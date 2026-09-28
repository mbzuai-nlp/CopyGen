"""CopyGen with SGLang 0.5.15.post1.

- decode_copygen: client-side CopyGen control loop on SGLang Engine responses
  (real hidden states and log-probs), used by sglang_client.py.
- CopyGenNativeWorker: a custom ``COPYGEN`` speculative algorithm that proposes
  the source continuation and runs the heads inside SGLang's TARGET_VERIFY step,
  used by sglang_native.py.
- HeadOverheadProcessor: custom logit processor that runs the heads on zero
  inputs and discards their outputs (used by head_overhead.py).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    move_accept_tokens_to_target_kvcache,
    record_stream_for_v2_verify,
)

from scripts.rebuttal.engine.common import SourceMatcher, load_heads, run_heads_on_zeros

# ---------------------------------------------------------------------------
# Client-side decoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenScore:
    token_id: int
    logprob: float


@dataclass
class NextTokenState:
    hidden: torch.Tensor
    top_scores: list[TokenScore]
    requested_scores: dict[int, float]


@dataclass
class ProposalScores:
    token_hidden: torch.Tensor
    token_logprobs: list[float]
    top_scores: list[list[TokenScore]]


@dataclass
class VerifiedProposal:
    start: int
    accepted_length: int
    margin_sum: float
    logprob_sum: float
    # Scores already returned at the first rejected position, reusable as the
    # next decision state when the proposal is accepted up to that point.
    boundary_state: NextTokenState | None = None


def _scores(value) -> list[TokenScore]:
    return [TokenScore(token_id=int(item[1]), logprob=float(item[0])) for item in value]


def _hidden_rows(meta: dict, hidden_size: int) -> torch.Tensor:
    # SGLang returns [prefill_matrix, decode_row, ...]; accept a bare matrix too.
    raw = meta["hidden_states"]
    first = raw[0]
    rows = first if isinstance(first[0], list) else raw
    tensor = torch.tensor(rows, dtype=torch.float32, device="cpu")
    if tensor.ndim != 2 or tensor.shape[1] != hidden_size:
        raise ValueError(f"unexpected hidden-state shape {tuple(tensor.shape)}")
    return tensor


class ScoreBackend:
    """Turns Engine.generate responses into the signals CopyGen needs."""

    def __init__(self, engine, hidden_size: int, next_top_k: int) -> None:
        self.engine = engine
        self.hidden_size = hidden_size
        self.next_top_k = next_top_k
        self.request_count = 0

    def _generate(self, input_ids, logprob_start_len, top_logprobs_num, token_ids_logprob=None):
        kwargs = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "top_k": 1,
                "max_new_tokens": 1,
                "ignore_eos": True,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "return_hidden_states": True,
        }
        if token_ids_logprob:
            kwargs["token_ids_logprob"] = token_ids_logprob
        self.request_count += 1
        return self.engine.generate(**kwargs)["meta_info"]

    def next_token(self, prefix_ids, requested_token_ids) -> NextTokenState:
        requested = list(dict.fromkeys(int(token_id) for token_id in requested_token_ids))
        meta = self._generate(
            prefix_ids, max(0, len(prefix_ids) - 1), self.next_top_k, requested or None
        )
        requested_scores = {}
        if requested:
            requested_scores = {
                score.token_id: score.logprob
                for score in _scores(meta["output_token_ids_logprobs"][0])
            }
        return NextTokenState(
            hidden=_hidden_rows(meta, self.hidden_size)[-1].clone(),
            top_scores=_scores(meta["output_top_logprobs"][0]),
            requested_scores=requested_scores,
        )

    def score_proposal(self, prefix_ids, proposal_ids) -> ProposalScores:
        """Prefill prefix + proposal and read the scores of every proposal token."""
        meta = self._generate([*prefix_ids, *proposal_ids], max(0, len(prefix_ids) - 1), 2)
        length = len(proposal_ids)
        return ProposalScores(
            token_hidden=_hidden_rows(meta, self.hidden_size)[-length:].clone(),
            token_logprobs=[
                score.logprob for score in _scores(meta["input_token_logprobs"][-length:])
            ],
            top_scores=[_scores(item) for item in meta["input_top_logprobs"][-length:]],
        )


def _to_heads(tensor: torch.Tensor, heads) -> torch.Tensor:
    parameter = next(heads.parameters())
    return tensor.to(device=parameter.device, dtype=parameter.dtype)


def _alternative(token_id: int, top_scores: list[TokenScore]) -> TokenScore:
    return top_scores[1] if top_scores[0].token_id == int(token_id) else top_scores[0]


def _score_of(state: NextTokenState, token_id: int) -> float:
    for score in state.top_scores:
        if score.token_id == token_id:
            return score.logprob
    return state.requested_scores[token_id]


def _verify_rows(heads, predictor_hidden, token_ids, token_logprobs, top_scores, margin_floor):
    """Learned-bias margins for a run of proposal tokens; returns the accepted prefix."""
    alternatives = [
        _alternative(token_id, scores) for token_id, scores in zip(token_ids, top_scores)
    ]
    raw_margins = [
        float(logprob - alternative.logprob)
        for logprob, alternative in zip(token_logprobs, alternatives)
    ]
    parameter = next(heads.parameters())
    source_bias = (
        heads.source_token_bias(
            _to_heads(predictor_hidden, heads),
            torch.tensor(raw_margins, dtype=parameter.dtype, device=parameter.device),
        )
        .float()
        .cpu()
    )
    adjusted = torch.tensor(raw_margins, dtype=torch.float32) + source_bias
    accepted = 0
    for margin in adjusted.tolist():
        if margin > margin_floor:
            accepted += 1
        else:
            break
    return accepted, float(adjusted[:accepted].sum().item()), float(sum(token_logprobs[:accepted]))


@torch.inference_mode()
def verify_candidate(
    backend,
    heads,
    next_state,
    prefix_ids,
    source_ids,
    start,
    proposed_length,
    margin_floor,
    chunk_size,
):
    """Verify one source proposal in stages.

    Token 0 is scored from the next-token response already in hand. If it
    passes, one prefill verifies token 0 plus up to ``chunk_size`` following
    tokens, and the remaining suffix is verified only if that stage fully passes.
    """
    proposal_ids = [int(token_id) for token_id in source_ids[start : start + proposed_length]]
    if not proposal_ids:
        return None
    first = proposal_ids[0]
    accepted, margin_sum, logprob_sum = _verify_rows(
        heads,
        next_state.hidden.reshape(1, -1),
        [first],
        [_score_of(next_state, first)],
        [next_state.top_scores],
        margin_floor,
    )
    if not accepted:
        return VerifiedProposal(start, 0, float("-inf"), float("-inf"))
    accepted_total = 1
    tail_hidden = None
    while accepted_total < len(proposal_ids):
        if tail_hidden is None:
            request_start, stage_end = 0, min(len(proposal_ids), 1 + chunk_size)
        else:
            request_start, stage_end = accepted_total, len(proposal_ids)
        scores = backend.score_proposal(
            [*prefix_ids, *proposal_ids[:request_start]], proposal_ids[request_start:stage_end]
        )
        skip = accepted_total - request_start
        token_ids = proposal_ids[request_start:stage_end][skip:]
        if skip == 0:
            predictor_hidden = torch.cat([tail_hidden, scores.token_hidden[:-1]], dim=0)
        else:
            predictor_hidden = scores.token_hidden[skip - 1 : -1]
        accepted, margin_part, logprob_part = _verify_rows(
            heads,
            predictor_hidden,
            token_ids,
            scores.token_logprobs[skip:],
            scores.top_scores[skip:],
            margin_floor,
        )
        accepted_total += accepted
        margin_sum += margin_part
        logprob_sum += logprob_part
        if accepted < len(token_ids):
            rejected = token_ids[accepted]
            boundary = NextTokenState(
                hidden=predictor_hidden[accepted].detach().clone(),
                top_scores=list(scores.top_scores[skip:][accepted]),
                requested_scores={rejected: float(scores.token_logprobs[skip:][accepted])},
            )
            return VerifiedProposal(start, accepted_total, margin_sum, logprob_sum, boundary)
        tail_hidden = scores.token_hidden[-1:].clone()
    return VerifiedProposal(start, accepted_total, margin_sum, logprob_sum)


@torch.inference_mode()
def rerank_fallback(heads, next_state: NextTokenState, source_ids, candidate_starts) -> int:
    """GEN step after a rejected COPY: rerank source first tokens with their learned bias."""
    candidate_ids = list(
        dict.fromkeys(
            int(source_ids[start]) for start in candidate_starts if 0 <= start < len(source_ids)
        )
    )
    if not candidate_ids:
        return next_state.top_scores[0].token_id
    raw_scores = {item.token_id: item.logprob for item in next_state.top_scores}
    raw_scores.update(next_state.requested_scores)
    top1, top2 = next_state.top_scores[0], next_state.top_scores[1]
    raw_margins = [
        raw_scores[token_id] - (top2.logprob if token_id == top1.token_id else top1.logprob)
        for token_id in candidate_ids
    ]
    parameter = next(heads.parameters())
    biases = (
        heads.source_token_bias(
            _to_heads(next_state.hidden.reshape(1, -1).expand(len(candidate_ids), -1), heads),
            torch.tensor(raw_margins, dtype=parameter.dtype, device=parameter.device),
        )
        .float()
        .cpu()
        .tolist()
    )
    reranked = dict(raw_scores)
    for token_id, raw_margin, bias in zip(candidate_ids, raw_margins, biases):
        alternative = top2 if token_id == top1.token_id else top1
        reranked[token_id] = float(alternative.logprob + float(raw_margin + bias))
    return int(max(reranked, key=lambda token_id: (reranked[token_id], -token_id)))


@torch.inference_mode()
def decode_copygen(
    backend,
    heads,
    tokenizer,
    prompt_ids,
    source_ids,
    matcher,
    max_steps=4096,
    max_output_tokens=4096,
    max_candidate_starts=8,
    fallback_lm_token_topk=4,
    min_match_length=5,
    margin_floor=0.0,
    chunk_size=17,
):
    """CopyGen decoding (trigger -> length -> verified COPY or calibrated GEN)."""
    eos_id = tokenizer.eos_token_id
    output_ids: list[int] = []
    prefix_ids = list(prompt_ids)
    copy_lengths: list[int] = []
    stop_reason = "max_steps"
    carried: NextTokenState | None = None
    for _ in range(max_steps):
        if len(output_ids) >= max_output_tokens:
            stop_reason = "max_output_tokens"
            break
        match_length = matcher.current_features().longest_match_length
        aligned_starts = []
        if match_length >= min_match_length:
            aligned_starts = matcher.candidate_next_starts(
                max_candidates=max_candidate_starts, longest_only=True
            )
        requested_ids = list(
            dict.fromkeys(
                int(source_ids[start]) for start in aligned_starts if 0 <= start < len(source_ids)
            )
        )

        # Reuse the scores returned at a partial-proposal boundary when they are
        # enough for this decision: no heads needed, or every aligned source
        # token is scored and top-2 provably decides the fallback rerank.
        next_state = None
        if carried is not None:
            top1, top2 = carried.top_scores[:2]
            reusable = match_length < min_match_length or (
                bool(aligned_starts)
                and set(requested_ids).issubset(carried.requested_scores)
                and top1.logprob > top2.logprob
                and top1.token_id not in set(requested_ids)
            )
            if reusable:
                next_state = carried
            carried = None
        if next_state is None:
            next_state = backend.next_token(prefix_ids, requested_ids)
        if not bool(torch.count_nonzero(next_state.hidden).item()):
            raise ValueError("SGLang returned an all-zero hidden state")

        selected = next_state.top_scores[0].token_id
        if match_length >= min_match_length:
            hidden = _to_heads(next_state.hidden, heads)
            trigger = heads.trigger_logits(hidden, match_length).float().cpu().reshape(-1)
            if int(torch.argmax(trigger).item()) == 1:
                proposed_length = int(
                    heads.predict_copy_length(hidden, match_length).reshape(-1)[0].item()
                )
                candidate_starts = list(aligned_starts)
                if not candidate_starts and fallback_lm_token_topk > 0:
                    fallback_tokens = [
                        item.token_id for item in next_state.top_scores[:fallback_lm_token_topk]
                    ]
                    candidate_starts = matcher.fallback_candidate_starts(
                        fallback_tokens, max_candidates=max_candidate_starts
                    )
                proposals = []
                for start in candidate_starts:
                    proposal = verify_candidate(
                        backend,
                        heads,
                        next_state,
                        prefix_ids,
                        source_ids,
                        start,
                        proposed_length,
                        margin_floor,
                        chunk_size,
                    )
                    if proposal is not None:
                        proposals.append(proposal)
                best = max(
                    proposals,
                    key=lambda item: (
                        float(item.accepted_length),
                        item.margin_sum,
                        item.logprob_sum,
                        -item.start,
                    ),
                    default=None,
                )
                if best is not None and best.accepted_length > 0:
                    length = min(best.accepted_length, max_output_tokens - len(output_ids))
                    accepted_ids = [
                        int(token_id) for token_id in source_ids[best.start : best.start + length]
                    ]
                    output_ids.extend(accepted_ids)
                    prefix_ids.extend(accepted_ids)
                    matcher.extend(accepted_ids)
                    copy_lengths.append(len(accepted_ids))
                    if length == best.accepted_length:
                        carried = best.boundary_state
                    continue
                selected = rerank_fallback(heads, next_state, source_ids, candidate_starts)

        if selected == eos_id:
            stop_reason = "eos"
            break
        output_ids.append(int(selected))
        prefix_ids.append(int(selected))
        matcher.append(int(selected))
    return {
        "generated_token_ids": output_ids,
        "copy_lengths": copy_lengths,
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Native speculative worker
# ---------------------------------------------------------------------------


def _mismatch_token(top_token_id: int) -> int:
    """A token id guaranteed to differ from the target's top-1 (forces rejection)."""
    return 1 if top_token_id == 0 else 0


@dataclass
class _RequestState:
    matcher: Any
    consumed_output_tokens: int = 0


@dataclass
class _RoundRow:
    rid: str
    match_length: int
    source_start: Optional[int]
    proposal_length: int
    predicted_length: int = 0
    trigger_copy: bool = False


class CopyGenNativeWorker(NGRAMWorker):
    """Greedy COPYGEN speculative worker on SGLang's TARGET_VERIFY path.

    Every decode round drafts ``root + source continuation`` from the first
    aligned source start (linear tree). After target verification the trigger,
    length and source-bias heads decide how much of the draft may be accepted;
    rejected positions are forced to mismatch the target's top-1.

    Settings come from environment variables set by sglang_native.py:
    COPYGEN_SGLANG_PLUGIN (plugin.pt), COPYGEN_SGLANG_SOURCES (JSON {rid:
    source_ids}) and COPYGEN_SGLANG_STATS (head timing written on cache flush).
    """

    def __init__(
        self,
        server_args,
        gpu_id,
        tp_rank,
        dp_rank,
        moe_ep_rank,
        attn_cp_rank,
        moe_dp_rank,
        nccl_port,
        target_worker,
    ):
        self.server_args = server_args
        self.enable_overlap = False
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = int(tp_rank)
        self.page_size = server_args.page_size
        self.draft_token_num = int(server_args.speculative_num_draft_tokens)
        self.speculative_num_draft_tokens = self.draft_token_num
        self.topk = 1
        self.speculative_num_steps = int(server_args.speculative_num_steps)
        self.max_trie_depth = int(server_args.speculative_ngram_max_trie_depth)
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.adaptive_controller = None
        self._prev_decode_rids: set[str] = set()
        sources = json.loads(Path(os.environ["COPYGEN_SGLANG_SOURCES"]).read_text(encoding="utf-8"))
        self._source_ids = {rid: tuple(map(int, ids)) for rid, ids in sources.items()}
        self._states: dict[str, _RequestState] = {}
        self._round_rows: list[_RoundRow] = []
        self._tokenizer = target_worker.tokenizer
        self._vocab_size = int(self.model_runner.model_config.vocab_size)
        plugin_path = os.environ["COPYGEN_SGLANG_PLUGIN"]
        checkpoint = torch.load(plugin_path, map_location="cpu", weights_only=True)
        self.min_match_length = int(checkpoint.get("min_match_length", 5))
        self.heads = load_heads(plugin_path, torch.device(self.device))
        self._reset_head_timing()

    def _reset_head_timing(self) -> None:
        self._head_calls = {"trigger": 0, "length": 0, "bias": 0}
        self._head_cuda_ms = {"trigger": 0.0, "length": 0.0, "bias": 0.0}
        self._pending_head_events = []

    def clear_cache_pool(self):
        stats_path = os.environ.get("COPYGEN_SGLANG_STATS")
        if stats_path and self.tp_rank == 0:
            Path(stats_path).write_text(
                json.dumps({"head_calls": self._head_calls, "head_cuda_ms": self._head_cuda_ms}),
                encoding="utf-8",
            )
        self._states.clear()
        self._prev_decode_rids.clear()
        self._reset_head_timing()

    def add_external_corpus(self, *_args, **_kwargs):
        raise RuntimeError("COPYGEN does not use an n-gram corpus")

    def commit_corpus_load(self, *_args, **_kwargs) -> None:
        raise RuntimeError("COPYGEN does not use an n-gram corpus")

    def remove_external_corpus(self, *_args, **_kwargs) -> None:
        return None

    def list_external_corpora(self) -> dict[str, int]:
        return {}

    def _update_ngram_corpus(self, _batch) -> None:
        return None

    def _prepare_draft_tokens(self, batch):
        drafts, masks, rows = [], [], []
        pad_token_id = self._tokenizer.eos_token_id or 0
        stride = self.draft_token_num
        mask = np.tril(np.ones((stride, stride), dtype=np.bool_))
        for req in batch.reqs:
            rid = str(req.rid)
            output_ids = [int(token_id) for token_id in req.output_ids]
            state = self._states.get(rid)
            if state is None:
                state = _RequestState(
                    SourceMatcher(self._source_ids[rid], tokenizer=self._tokenizer)
                )
                self._states[rid] = state
            if state.consumed_output_tokens > len(output_ids):
                state.matcher.reset()
                state.consumed_output_tokens = 0
            state.matcher.extend(output_ids[state.consumed_output_tokens :])
            state.consumed_output_tokens = len(output_ids)
            match_length = int(state.matcher.current_features().longest_match_length)
            start = None
            if match_length >= self.min_match_length:
                starts = state.matcher.candidate_next_starts(max_candidates=8, longest_only=True)
                if starts:
                    start = starts[0]
            source_ids = self._source_ids[rid]
            continuation = [] if start is None else list(source_ids[start : start + stride - 1])
            tokens = [output_ids[-1], *continuation]
            tokens.extend([int(pad_token_id)] * (stride - len(tokens)))
            drafts.extend(tokens)
            masks.append(mask)
            rows.append(
                _RoundRow(
                    rid=rid,
                    match_length=match_length,
                    source_start=start,
                    proposal_length=len(continuation) if start is not None else 0,
                )
            )
        self._round_rows = rows
        return np.asarray(drafts, dtype=np.int64), np.stack(masks, axis=0).reshape(-1)

    def _force_rejection(self, candidates, logits, row_index: int, position: int = 1) -> None:
        top = int(torch.argmax(logits[row_index, position - 1]).item())
        candidates[row_index, position] = _mismatch_token(top)

    def _timed_head(self, name: str, operation):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = operation()
        end.record()
        self._pending_head_events.append((name, start, end))
        return value

    def _settle_head_timing(self) -> None:
        if not self._pending_head_events:
            return
        torch.cuda.synchronize()
        for name, start, end in self._pending_head_events:
            self._head_cuda_ms[name] += float(start.elapsed_time(end))
        self._pending_head_events.clear()

    @torch.inference_mode()
    def _apply_copygen_policy(self, verify_input, logits_output) -> None:
        bs = len(self._round_rows)
        stride = self.draft_token_num
        logits = logits_output.next_token_logits.reshape(bs, stride, self._vocab_size)
        hidden = logits_output.hidden_states.reshape(bs, stride, -1)
        candidates = verify_input.draft_token.reshape(bs, stride)
        dtype = next(self.heads.parameters()).dtype

        eligible = []
        for row_index, row in enumerate(self._round_rows):
            if row.source_start is None or row.proposal_length < 1:
                self._force_rejection(candidates, logits, row_index)
                continue
            trigger = self._timed_head(
                "trigger",
                lambda: self.heads.trigger_logits(
                    hidden[row_index, 0:1].to(dtype), row.match_length
                ),
            )
            self._head_calls["trigger"] += 1
            row.trigger_copy = int(torch.argmax(trigger, dim=-1).item()) == 1
            if not row.trigger_copy:
                self._force_rejection(candidates, logits, row_index)
                continue
            predicted = self._timed_head(
                "length",
                lambda: self.heads.predict_copy_length(
                    hidden[row_index, 0:1].to(dtype), row.match_length
                ),
            )
            self._head_calls["length"] += 1
            row.predicted_length = min(int(predicted.item()), row.proposal_length, stride - 1)
            eligible.append(row_index)

        for row_index in eligible:
            length = self._round_rows[row_index].predicted_length
            # Draft node j is predicted by node j - 1 (linear tree).
            child_tensor = torch.arange(1, length + 1, dtype=torch.long, device=candidates.device)
            parent_tensor = torch.arange(0, length, dtype=torch.long, device=logits.device)
            source_tokens = candidates[row_index, child_tensor].long()
            row_logits = logits[row_index, parent_tensor]
            source_logits = row_logits.gather(1, source_tokens.unsqueeze(1)).squeeze(1)
            top_values, top_ids = torch.topk(row_logits, k=2, dim=-1)
            alternatives = torch.where(
                top_ids[:, 0].eq(source_tokens), top_values[:, 1], top_values[:, 0]
            )
            raw_margins = source_logits - alternatives
            biases = self._timed_head(
                "bias",
                lambda: self.heads.source_token_bias(
                    hidden[row_index, parent_tensor].to(dtype), raw_margins.to(dtype)
                ),
            ).to(row_logits.dtype)
            self._head_calls["bias"] += 1
            logits[row_index, parent_tensor, source_tokens] += biases
            if length < self._round_rows[row_index].proposal_length:
                # Stop acceptance after the predicted copy length.
                self._force_rejection(candidates, logits, row_index, position=length + 1)

    def forward_batch_generation(self, batch, on_publish=None):
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)
        bs = len(batch.reqs)
        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        verify_input = batch.spec_info
        accept_lens = torch.ones(bs, dtype=torch.int32, device=self.device)
        if batch.forward_mode.is_target_verify():
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_result = self.target_worker.forward_batch_generation(batch, is_verify=True)
            logits_output = batch_result.logits_output
            can_run_cuda_graph = batch_result.can_run_cuda_graph
            self._apply_copygen_policy(verify_input, logits_output)
            predict, native_accept_lens, accept_index = eagle_sample(
                verify_input, batch, logits_output, None
            )
            accepted_drafts = native_accept_lens - 1
            # SGLang appends a bonus token after the accepted drafts; a COPY
            # action emits only the copied tokens.
            emitted = []
            for index in range(bs):
                native = int(native_accept_lens[index].item())
                copy_action = self._round_rows[index].trigger_copy
                emitted.append(native - 1 if copy_action and native > 1 else native)
            accept_lens = torch.tensor(emitted, dtype=torch.int32, device=self.device)
            new_seq_lens = batch.seq_lens + accept_lens
            commit_mamba_states_after_verify(
                self.target_worker, batch, accept_lens, accept_index, self.draft_token_num
            )
            accept_tokens = predict[accept_index].flatten()
            next_token_ids = accept_tokens
            move_accept_tokens_to_target_kvcache(
                batch, accept_index, accepted_drafts, self.token_to_kv_pool_allocator
            )
            if batch.return_logprob:
                compute_spec_v2_logprobs(
                    batch, logits_output, predict, accept_index, self.draft_token_num - 1
                )
            if on_publish is not None:
                on_publish(new_seq_lens)
            self._settle_head_timing()
            batch.forward_mode = ForwardMode.DECODE
        else:
            batch_result = self.target_worker.forward_batch_generation(batch)
            logits_output = batch_result.logits_output
            predict = batch_result.next_token_ids
            can_run_cuda_graph = batch_result.can_run_cuda_graph
            new_seq_lens = batch.seq_lens.clone()
            accept_tokens = torch.zeros(
                bs, self.draft_token_num, dtype=torch.int32, device=self.device
            )
            accept_tokens[:, 0] = predict
            accept_tokens = accept_tokens.flatten()
            next_token_ids = predict
            if on_publish is not None:
                on_publish(new_seq_lens)

        next_draft_input = NgramVerifyInput(
            draft_token_num=self.draft_token_num,
            new_seq_lens=new_seq_lens,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
        )


_REGISTERED = False


def register_copygen_algorithm() -> None:
    """Register ``COPYGEN`` as a custom SGLang speculative algorithm."""
    global _REGISTERED
    if _REGISTERED:
        return
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import CustomSpecAlgo

    class CopyGenSpecAlgo(CustomSpecAlgo):
        def is_ngram(self) -> bool:
            return True

        def has_draft_kv(self) -> bool:
            return False

        def carries_draft_hidden_states(self) -> bool:
            return False

        def need_topk(self) -> bool:
            return False

        def create_future_map(self, device, req_to_token_pool, needs_cpu_seq_lens: bool = True):
            from sglang.srt.managers.overlap_utils import FutureMap

            return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)

        def handle_server_args(self, server_args) -> None:
            from sglang.srt.arg_groups.speculative_hook import _handle_ngram

            _handle_ngram(server_args)
            server_args.disable_overlap_schedule = True
            server_args.speculative_eagle_topk = 1

    @SpeculativeAlgorithm.register("COPYGEN", supports_overlap=False, spec_class=CopyGenSpecAlgo)
    def _worker_factory(_server_args):
        return CopyGenNativeWorker

    _REGISTERED = True


# ---------------------------------------------------------------------------
# Head-overhead logit processor
# ---------------------------------------------------------------------------


class HeadOverheadProcessor(CustomLogitProcessor):
    """Runs the trained heads on zero inputs; token logits are left unchanged.

    Reads COPYGEN_SGLANG_HEADS (JSON: plugins, candidate_items, max_batch_rows).
    """

    def __init__(self) -> None:
        self._heads = None

    def _initialize(self, logits: torch.Tensor) -> None:
        config = json.loads(os.environ["COPYGEN_SGLANG_HEADS"])
        device = logits.device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self._heads = {
            domain: load_heads(path, device, dtype)
            for domain, path in sorted(config["plugins"].items())
        }
        hidden_size = next(iter(self._heads.values())).plugin_input_norm.weight.numel() - 1
        self._candidate_items = int(config["candidate_items"])
        rows = int(config["max_batch_rows"])
        items = rows * self._candidate_items
        self._hidden = torch.zeros((rows, hidden_size), dtype=dtype, device=device)
        self._matches = torch.full((rows,), 5.0, dtype=dtype, device=device)
        self._bias_hidden = torch.zeros((items, hidden_size), dtype=dtype, device=device)
        self._margins = torch.zeros((items,), dtype=dtype, device=device)

    @torch.inference_mode()
    def __call__(
        self, logits: torch.Tensor, custom_param_list: list[dict] | None = None
    ) -> torch.Tensor:
        if self._heads is None:
            self._initialize(logits)
        grouped = defaultdict(list)
        for row_index, params in enumerate(custom_param_list):
            grouped[(params["copygen_domain"], params["copygen_mode"])].append(row_index)
        for (domain, mode), row_indices in grouped.items():
            if mode == "noop":
                continue
            count = len(row_indices)
            items = count * self._candidate_items
            sink = run_heads_on_zeros(
                self._heads[domain],
                mode,
                self._hidden[:count],
                self._matches[:count],
                self._bias_hidden[:items],
                self._margins[:items],
            )
            rows = torch.tensor(row_indices, dtype=torch.long, device=logits.device)
            columns = torch.zeros_like(rows)
            logits.index_put_((rows, columns), logits[rows, columns] + sink.to(logits.dtype) * 0.0)
        return logits
