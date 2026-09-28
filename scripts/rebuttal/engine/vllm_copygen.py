"""CopyGen inside vLLM 0.25.1 (V1 engine, in-process).

- CopyGenWorker: speculative-decoding worker. The trained heads run on the
  target hidden states; an accepted COPY decision drafts the source
  continuation, and the learned source bias is added during verification.
- ProbeWorker: returns target logits/hidden-state evidence for the
  client-controlled decoder in vllm_full_decode.py.
- HeadOverheadProcessor: logits processor that runs the heads on zero inputs
  and discards their outputs (used by head_overhead.py).

The workers read their settings from the COPYGEN_VLLM_CONFIG environment
variable (JSON) because vLLM constructs them itself.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from scripts.rebuttal.engine.common import SourceMatcher, load_heads, run_heads_on_zeros

CONFIG_ENV = "COPYGEN_VLLM_CONFIG"

# SamplingParams.extra_args keys.
MODE_ARG = "copygen_mode"
DOMAIN_ARG = "copygen_domain"
SOURCE_IDS_ARG = "copygen_source_ids"
PROBE_KEY_ARG = "copygen_probe_key"
PROBE_KIND_ARG = "copygen_probe_kind"
MATCH_LENGTH_ARG = "copygen_match_length"
ALIGNED_FIRST_IDS_ARG = "copygen_aligned_first_token_ids"
PREFIX_LENGTH_ARG = "copygen_prefix_length"
PROPOSAL_IDS_ARG = "copygen_proposal_ids"
WINDOW_START_ARG = "copygen_window_start"
WINDOW_END_ARG = "copygen_window_end"


def _config() -> dict:
    return json.loads(os.environ[CONFIG_ENV])


def _swap_runner_class(worker: Worker, runner_class) -> None:
    import vllm.v1.worker.gpu_model_runner as runner_module

    default_runner = runner_module.GPUModelRunner
    runner_module.GPUModelRunner = runner_class
    try:
        Worker.init_device(worker)
    finally:
        runner_module.GPUModelRunner = default_runner


def _raw_margins(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Source-token logit minus the best competing logit, row by row."""
    top_values, top_ids = torch.topk(logits, k=2, dim=-1)
    source_logits = logits.gather(1, token_ids.reshape(-1, 1)).squeeze(1)
    alternatives = torch.where(top_ids[:, 0] == token_ids, top_values[:, 1], top_values[:, 0])
    return source_logits - alternatives


class NoopProposer:
    """Custom-class proposer required by vLLM; CopyGenWorker supplies the drafts."""

    def __init__(self, vllm_config) -> None:
        self.max_tokens = int(vllm_config.speculative_config.num_speculative_tokens)

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *, slot_mappings=None):
        return [[] for _ in sampled_token_ids]


# ---------------------------------------------------------------------------
# Speculative-decoding worker
# ---------------------------------------------------------------------------


@dataclass
class _MatcherState:
    source_ids: list[int]
    matcher: SourceMatcher
    consumed_output_tokens: int = 0


@dataclass
class _Candidate:
    req_id: str
    row_index: int
    state: _MatcherState
    match_length: int
    predicted_length: int = 0


@dataclass(frozen=True)
class _SpecLayout:
    req_id: str
    request_index: int
    draft_start: int
    draft_count: int
    bonus_logits_index: int


def _spec_layouts(req_ids: list[str], num_draft_tokens: list[int]) -> list[_SpecLayout]:
    """Position of every request in vLLM's flattened speculative tensors."""
    layouts = []
    draft_offset = 0
    logits_offset = 0
    for index, (req_id, count) in enumerate(zip(req_ids, num_draft_tokens)):
        layouts.append(_SpecLayout(req_id, index, draft_offset, count, logits_offset + count))
        draft_offset += count
        logits_offset += count + 1
    return layouts


class CopyGenModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device) -> None:
        super().__init__(vllm_config, device)
        config = _config()
        self.copygen_config = config
        self.min_match_length = int(config["min_match_length"])
        self.max_candidates = int(config["max_candidate_starts"])
        self.margin_floor = float(config["verifier_margin_floor"])
        self.copygen_heads = {}
        self.copygen_matchers: dict[str, _MatcherState] = {}
        self.copygen_pending: dict[str, dict] = {}
        self.copygen_hidden = None

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        for domain, path in sorted(self.copygen_config["plugins"].items()):
            self.copygen_heads[domain] = load_heads(path, self.device)
        # Warm the head kernels once so the first timed call is not a cold start.
        hidden_size = int(self.model_config.hf_config.hidden_size)
        with torch.inference_mode():
            hidden = torch.zeros((1, hidden_size), dtype=torch.bfloat16, device=self.device)
            margin = torch.zeros(1, dtype=torch.bfloat16, device=self.device)
            for heads in self.copygen_heads.values():
                for _ in range(int(self.copygen_config["head_warmup_iterations"])):
                    heads.trigger_logits(hidden, self.min_match_length)
                    heads.predict_copy_length(hidden, self.min_match_length)
                    heads.source_token_bias(hidden, margin)
            torch.cuda.synchronize(self.device)

    def _request_args(self, req_id: str):
        request = self.requests.get(req_id)
        if request is None or request.sampling_params is None:
            return None, None
        return request, request.sampling_params.extra_args or {}

    def _sync_matcher(self, req_id: str, request, extra: dict) -> _MatcherState:
        state = self.copygen_matchers.get(req_id)
        if state is None:
            source_ids = [int(token_id) for token_id in extra[SOURCE_IDS_ARG]]
            state = _MatcherState(source_ids, SourceMatcher(source_ids))
            self.copygen_matchers[req_id] = state
        output_ids = request.output_token_ids
        if state.consumed_output_tokens > len(output_ids):
            state.matcher.reset()
            state.consumed_output_tokens = 0
        if state.consumed_output_tokens < len(output_ids):
            state.matcher.extend(output_ids[state.consumed_output_tokens :])
            state.consumed_output_tokens = len(output_ids)
        return state

    def _copy_decisions(self, logits, hidden, row_specs) -> dict[str, dict]:
        """Run the heads for ordinary rows; bias the chosen source first token."""
        candidates: dict[str, list[_Candidate]] = {}
        discard = self.discard_request_mask.np
        for req_id, request_index, row_index in row_specs:
            if bool(discard[request_index]):
                continue
            request, extra = self._request_args(req_id)
            if request is None or extra.get(MODE_ARG) != "copygen":
                continue
            state = self._sync_matcher(req_id, request, extra)
            match_length = int(state.matcher.current_features().longest_match_length)
            if match_length < self.min_match_length:
                continue
            domain = extra[DOMAIN_ARG]
            candidates.setdefault(domain, []).append(
                _Candidate(req_id, row_index, state, match_length)
            )

        decisions: dict[str, dict] = {}
        max_draft = int(self.speculative_config.num_speculative_tokens)
        for domain, items in candidates.items():
            heads = self.copygen_heads[domain]
            rows = torch.tensor(
                [item.row_index for item in items], dtype=torch.long, device=hidden.device
            )
            matches = torch.tensor(
                [item.match_length for item in items], dtype=torch.bfloat16, device=hidden.device
            )
            grouped_hidden = hidden[rows]
            choices = heads.trigger_logits(grouped_hidden, matches).argmax(dim=-1).cpu().tolist()
            positions = [index for index, choice in enumerate(choices) if int(choice) == 1]
            if not positions:
                continue
            triggered = [items[index] for index in positions]
            position_tensor = torch.tensor(positions, dtype=torch.long, device=hidden.device)
            lengths = heads.predict_copy_length(
                grouped_hidden[position_tensor], matches[position_tensor]
            )

            proposed: list[_Candidate] = []
            offsets = []
            flat_rows, flat_positions, flat_ids, flat_starts = [], [], [], []
            for item, length in zip(triggered, lengths.reshape(-1).cpu().tolist()):
                item.predicted_length = int(length)
                starts = item.state.matcher.candidate_next_starts(
                    max_candidates=self.max_candidates, longest_only=True
                )
                starts = [
                    int(start) for start in starts if 0 <= int(start) < len(item.state.source_ids)
                ]
                if not starts or item.predicted_length < 1:
                    continue
                begin = len(flat_rows)
                flat_rows.extend([item.row_index] * len(starts))
                flat_positions.extend([len(proposed)] * len(starts))
                flat_ids.extend(item.state.source_ids[start] for start in starts)
                flat_starts.extend(starts)
                offsets.append((begin, len(flat_rows)))
                proposed.append(item)
            if not proposed:
                continue

            device = logits.device
            request_rows = torch.tensor(
                [item.row_index for item in proposed], dtype=torch.long, device=device
            )
            candidate_positions = torch.tensor(flat_positions, dtype=torch.long, device=device)
            candidate_ids = torch.tensor(flat_ids, dtype=torch.long, device=device)
            # One top-2 per request row, shared by all of its candidate starts.
            top_values, top_ids = torch.topk(logits[request_rows], k=2, dim=-1)
            source_values = logits[request_rows[candidate_positions], candidate_ids]
            request_top_ids = top_ids[candidate_positions]
            request_top_values = top_values[candidate_positions]
            alternatives = torch.where(
                request_top_ids[:, 0] == candidate_ids,
                request_top_values[:, 1],
                request_top_values[:, 0],
            )
            raw_margins = source_values - alternatives
            flat_row_tensor = torch.tensor(flat_rows, dtype=torch.long, device=device)
            source_bias = heads.source_token_bias(hidden[flat_row_tensor], raw_margins)
            adjusted = (raw_margins + source_bias.float()).cpu().tolist()

            biased_rows, biased_ids, bias_positions = [], [], []
            for item, (begin, end) in zip(proposed, offsets):
                local = adjusted[begin:end]
                best = begin + max(range(len(local)), key=local.__getitem__)
                if float(adjusted[best]) < self.margin_floor:
                    continue
                start = flat_starts[best]
                first_token = item.state.source_ids[start]
                length = min(
                    item.predicted_length, len(item.state.source_ids) - start, max_draft + 1
                )
                decisions[item.req_id] = {
                    "first_token": first_token,
                    "continuation": item.state.source_ids[start + 1 : start + length],
                }
                biased_rows.append(item.row_index)
                biased_ids.append(first_token)
                bias_positions.append(best)
            if biased_rows:
                logits[
                    torch.tensor(biased_rows, dtype=torch.long, device=device),
                    torch.tensor(biased_ids, dtype=torch.long, device=device),
                ] += source_bias[torch.tensor(bias_positions, dtype=torch.long, device=device)].to(
                    logits.dtype
                )
        return decisions

    def _bias_drafts(self, logits, hidden, spec_decode_metadata, layouts) -> None:
        """Add the learned source bias to every drafted (copied) token."""
        target_indices = spec_decode_metadata.target_logits_indices.long()
        draft_ids = spec_decode_metadata.draft_token_ids.long()
        discard = self.discard_request_mask.np
        positions_by_domain: dict[str, list[int]] = {}
        for layout in layouts:
            if layout.draft_count == 0 or bool(discard[layout.request_index]):
                continue
            _, extra = self._request_args(layout.req_id)
            positions_by_domain.setdefault(extra[DOMAIN_ARG], []).extend(
                range(layout.draft_start, layout.draft_start + layout.draft_count)
            )
        for domain, positions in positions_by_domain.items():
            position_tensor = torch.tensor(
                positions, dtype=torch.long, device=target_indices.device
            )
            rows = target_indices[position_tensor]
            token_ids = draft_ids[position_tensor]
            raw_margins = _raw_margins(logits[rows], token_ids)
            source_bias = self.copygen_heads[domain].source_token_bias(hidden[rows], raw_margins)
            logits[rows, token_ids] += source_bias.to(logits.dtype)

    def _sample(self, logits, spec_decode_metadata):
        if logits is None or self.copygen_hidden is None:
            return GPUModelRunner._sample(self, logits, spec_decode_metadata)
        hidden = self.copygen_hidden
        req_ids = list(self.input_batch.req_ids)
        for req_id in req_ids:
            self.copygen_pending.pop(req_id, None)
        if spec_decode_metadata is None:
            row_specs = [(req_id, index, index) for index, req_id in enumerate(req_ids)]
        else:
            layouts = _spec_layouts(req_ids, list(spec_decode_metadata.num_draft_tokens))
            self._bias_drafts(logits, hidden, spec_decode_metadata, layouts)
            row_specs = [
                (layout.req_id, layout.request_index, layout.bonus_logits_index)
                for layout in layouts
                if layout.draft_count == 0
            ]
        decisions = self._copy_decisions(logits, hidden, row_specs)
        output = GPUModelRunner._sample(self, logits, spec_decode_metadata)
        self.copygen_pending.update(decisions)
        return output

    def sample_tokens(self, grammar_output):
        state = self.execute_model_state
        self.copygen_hidden = state.sample_hidden_states if state is not None else None
        try:
            return super().sample_tokens(grammar_output)
        finally:
            self.copygen_hidden = None

    def propose_draft_token_ids(
        self,
        scheduler_output,
        sampled_token_ids,
        sampling_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        spec_decode_metadata,
        common_attn_metadata,
        slot_mappings,
    ):
        max_to_schedule = int(scheduler_output.num_spec_tokens_to_schedule)
        req_ids = list(self.input_batch.req_ids)
        draft_counts = (
            list(spec_decode_metadata.num_draft_tokens)
            if spec_decode_metadata is not None
            else [0] * len(req_ids)
        )
        drafts = []
        for req_id, sampled_row, draft_count in zip(req_ids, sampled_token_ids, draft_counts):
            decision = self.copygen_pending.pop(req_id, None)
            sampled = int(sampled_row[0]) if sampled_row else None
            # Draft the source continuation only if the target emitted the
            # chosen source first token.
            if (
                draft_count == 0
                and decision is not None
                and sampled == decision["first_token"]
                and decision["continuation"]
            ):
                drafts.append(list(decision["continuation"][:max_to_schedule]))
            else:
                drafts.append([])
        return drafts

    def _update_states(self, scheduler_output):
        for req_id in scheduler_output.finished_req_ids:
            self.copygen_matchers.pop(req_id, None)
            self.copygen_pending.pop(req_id, None)
        return super()._update_states(scheduler_output)


class CopyGenWorker(Worker):
    def init_device(self) -> None:
        _swap_runner_class(self, CopyGenModelRunner)


# ---------------------------------------------------------------------------
# Probe worker for client-controlled decoding
# ---------------------------------------------------------------------------

_PROBE_LOCK = threading.Lock()
_PROBE_RESULTS: list[dict] = []


def _to_python(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    return value


def drain_probe_results() -> list[dict]:
    """Collect the evidence recorded during the last llm.generate call."""
    with _PROBE_LOCK:
        results = list(_PROBE_RESULTS)
        _PROBE_RESULTS.clear()
    if results:
        torch.cuda.synchronize()
    return [_to_python(result) for result in results]


def _record_probe(result: dict) -> None:
    with _PROBE_LOCK:
        _PROBE_RESULTS.append(result)


class ProbeModelRunner(GPUModelRunner):
    """Records CopyGen evidence for probe requests; sampling is unchanged.

    A state probe returns the top-9 target logits at the last prompt position,
    the trigger/length outputs, and the calibrated scores of the candidate
    source first tokens. A candidate probe scores a window of one source
    proposal: target logits, top-2 alternative, learned bias and log-prob.
    """

    def __init__(self, vllm_config, device) -> None:
        super().__init__(vllm_config, device)
        config = _config()
        self.copygen_config = config
        self.min_match_length = int(config["min_match_length"])
        self.fallback_topk = int(config["fallback_lm_token_topk"])
        self.state_topn = int(config["state_topn"])
        self.copygen_heads = {}
        self.probe_scheduler_output = None
        self.probe_sample_hidden = None

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        for domain, path in sorted(self.copygen_config["plugins"].items()):
            self.copygen_heads[domain] = load_heads(path, self.device)

    def _candidate_probes(self, state) -> None:
        scheduler_output = state.scheduler_output
        hidden_states = state.hidden_states[: scheduler_output.total_num_scheduled_tokens]
        for req_id in self.input_batch.req_ids:
            request = self.requests[req_id]
            extra = request.sampling_params.extra_args
            if extra[PROBE_KIND_ARG] != "candidate":
                continue
            prefix_length = extra[PREFIX_LENGTH_ARG]
            proposal_ids = extra[PROPOSAL_IDS_ARG]
            # Proposal token j is predicted by the hidden state at prompt
            # position prefix_length - 1 + j. Intersect the requested window
            # with the slice of the prompt scheduled in this step.
            desired_start = prefix_length - 1 + extra[WINDOW_START_ARG]
            desired_end = prefix_length - 1 + extra[WINDOW_END_ARG]
            scheduled_start = int(request.num_computed_tokens)
            scheduled_end = scheduled_start + int(scheduler_output.num_scheduled_tokens[req_id])
            begin = max(desired_start, scheduled_start)
            end = min(desired_end, scheduled_end)
            if begin >= end:
                continue
            offset_start = begin - (prefix_length - 1)
            offset_end = end - (prefix_length - 1)
            packed_start = int(self.query_start_loc.np[self.input_batch.req_id_to_index[req_id]])
            hidden = hidden_states[
                packed_start + begin - scheduled_start : packed_start + end - scheduled_start
            ]
            target_ids = torch.tensor(
                proposal_ids[offset_start:offset_end], dtype=torch.long, device=hidden.device
            )
            logits = self.model.compute_logits(hidden).float()
            raw_margins = _raw_margins(logits, target_ids)
            source_bias = (
                self.copygen_heads[extra[DOMAIN_ARG]].source_token_bias(hidden, raw_margins).float()
            )
            logprobs = F.log_softmax(logits, dim=-1).gather(1, target_ids.reshape(-1, 1)).squeeze(1)
            _record_probe(
                {
                    "probe_key": extra[PROBE_KEY_ARG],
                    "offset_start": offset_start,
                    "adjusted_margins": (raw_margins + source_bias).clone(),
                    "logprobs": logprobs.clone(),
                }
            )

    def _state_probe(self, extra: dict, row_logits, row_hidden) -> None:
        top_values, top_ids = torch.topk(
            row_logits, k=min(self.state_topn, row_logits.shape[-1]), dim=-1
        )
        result = {
            "probe_key": extra[PROBE_KEY_ARG],
            "top_ids": top_ids.clone(),
            "top_logits": top_values.clone(),
            "trigger_logits": None,
            "predicted_length": None,
            "fallback": None,
        }
        match_length = extra[MATCH_LENGTH_ARG]
        if match_length >= self.min_match_length:
            heads = self.copygen_heads[extra[DOMAIN_ARG]]
            trigger = heads.trigger_logits(row_hidden, match_length).float()
            result["trigger_logits"] = trigger[0].clone()
            if int(torch.argmax(trigger, dim=-1)[0].item()) == 1:
                # Score the aligned source first tokens, or the top-k LM tokens
                # when no source position is aligned.
                token_ids = list(extra[ALIGNED_FIRST_IDS_ARG])
                if not token_ids:
                    token_ids = list(dict.fromkeys(top_ids.cpu().tolist()[: self.fallback_topk]))
                token_tensor = torch.tensor(token_ids, dtype=torch.long, device=row_logits.device)
                source_logits = row_logits.index_select(0, token_tensor)
                top2_values, top2_ids = torch.topk(row_logits, k=2, dim=-1)
                source_is_top = token_tensor.eq(top2_ids[0])
                alternative_logits = torch.where(
                    source_is_top,
                    top2_values[1].expand_as(source_logits),
                    top2_values[0].expand_as(source_logits),
                )
                raw_margins = source_logits - alternative_logits
                source_bias = heads.source_token_bias(
                    row_hidden.expand(token_tensor.shape[0], -1), raw_margins
                ).float()
                result["predicted_length"] = (
                    heads.predict_copy_length(row_hidden, match_length).reshape(-1)[0].clone()
                )
                result["fallback"] = {
                    "token_ids": token_ids,
                    "alternative_logits": alternative_logits.clone(),
                    "adjusted_margins": (raw_margins + source_bias).clone(),
                }
        _record_probe(result)

    def _sample(self, logits, spec_decode_metadata):
        scheduler_output = self.probe_scheduler_output
        sample_hidden = self.probe_sample_hidden
        if (
            logits is None
            or spec_decode_metadata is not None
            or scheduler_output is None
            or sample_hidden is None
        ):
            return super()._sample(logits, spec_decode_metadata)
        for req_index, req_id in enumerate(self.input_batch.req_ids):
            request = self.requests[req_id]
            extra = request.sampling_params.extra_args
            if extra[PROBE_KIND_ARG] != "state":
                continue
            num_scheduled = int(scheduler_output.num_scheduled_tokens[req_id])
            if int(request.num_computed_tokens) + num_scheduled < len(request.prompt_token_ids):
                continue
            self._state_probe(
                extra,
                logits[req_index].float(),
                sample_hidden[req_index : req_index + 1],
            )
        return super()._sample(logits, spec_decode_metadata)

    @torch.inference_mode()
    def sample_tokens(self, grammar_output):
        state = self.execute_model_state
        if state is None:
            return super().sample_tokens(grammar_output)
        self.probe_scheduler_output = state.scheduler_output
        self.probe_sample_hidden = state.sample_hidden_states
        self._candidate_probes(state)
        try:
            return super().sample_tokens(grammar_output)
        finally:
            self.probe_scheduler_output = None
            self.probe_sample_hidden = None


class ProbeWorker(Worker):
    def init_device(self) -> None:
        _swap_runner_class(self, ProbeModelRunner)


# ---------------------------------------------------------------------------
# Head-overhead logits processor
# ---------------------------------------------------------------------------


class HeadOverheadProcessor(LogitsProcessor):
    """Runs the trained heads on zero inputs; token logits are left unchanged."""

    @classmethod
    def validate_params(cls, params) -> None:
        return None

    def __init__(self, vllm_config, device, is_pin_memory) -> None:
        config = _config()
        hidden_size = int(vllm_config.model_config.hf_config.hidden_size)
        self.candidate_items = int(config["candidate_items"])
        self.heads = {
            domain: load_heads(path, torch.device(device))
            for domain, path in sorted(config["plugins"].items())
        }
        max_requests = int(vllm_config.scheduler_config.max_num_seqs)
        items = max_requests * self.candidate_items
        self.hidden = torch.zeros((max_requests, hidden_size), dtype=torch.bfloat16, device=device)
        self.matches = torch.full((max_requests,), 5.0, dtype=torch.bfloat16, device=device)
        self.bias_hidden = torch.zeros((items, hidden_size), dtype=torch.bfloat16, device=device)
        self.margins = torch.zeros((items,), dtype=torch.bfloat16, device=device)
        self.req_info: dict[int, tuple[str, str]] = {}

    def is_argmax_invariant(self) -> bool:
        # Keep the same sampler path as a processor that does change logits.
        return False

    @staticmethod
    def _new_state(params, prompt_ids, output_ids):
        extra = params.extra_args or {}
        if MODE_ARG not in extra:
            return None
        return extra[DOMAIN_ARG], extra[MODE_ARG]

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.req_info, batch_update, self._new_state)

    @torch.inference_mode()
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        grouped = defaultdict(list)
        for row_index, state in self.req_info.items():
            grouped[state].append(row_index)
        for (domain, mode), row_indices in grouped.items():
            rows = torch.tensor(row_indices, dtype=torch.long, device=logits.device)
            if mode == "noop":
                continue
            count = len(row_indices)
            items = count * self.candidate_items
            sink = run_heads_on_zeros(
                self.heads[domain],
                mode,
                self.hidden[:count],
                self.matches[:count],
                self.bias_hidden[:items],
                self.margins[:items],
            )
            columns = torch.zeros_like(rows)
            logits.index_put_((rows, columns), logits[rows, columns] + sink.to(logits.dtype) * 0.0)
        return logits
