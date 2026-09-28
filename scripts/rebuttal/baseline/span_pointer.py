#!/usr/bin/env python3
"""
Frozen-LM span pointer (inspired by Copy That!, Panthaplackel et al., 2021).

Actions are Gen(v) from the frozen LM head or Copy(i:j) of a context span. A gate mixes the
two branches, and a span is scored as start_score[i] + end_score[j], so the normalizer over
all spans is computed in O(N) without an N x N matrix. Only the start/end/gate heads train.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def all_span_logsumexp(start_scores: torch.Tensor, end_scores: torch.Tensor) -> torch.Tensor:
    """logsumexp over all spans start <= end of start_scores[start] + end_scores[end]."""
    suffix_end = torch.logcumsumexp(end_scores.float().flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
    return torch.logsumexp(start_scores.float() + suffix_end, dim=-1)


def best_span(start_scores: torch.Tensor, end_scores: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Highest-scoring span (inclusive end) of one-dimensional start/end scores."""
    end_values, end_indices = torch.cummax(end_scores.float().flip(dims=(0,)), dim=0)
    suffix_end_values = end_values.flip(dims=(0,))
    suffix_end_indices = ((end_scores.numel() - 1) - end_indices).flip(dims=(0,))
    scores = start_scores.float() + suffix_end_values
    start = int(torch.argmax(scores).item())
    return scores[start], start, int(suffix_end_indices[start].item())


class SpanPointerModel(nn.Module):
    def __init__(self, model_path: str, score_scale: float | None = None) -> None:
        super().__init__()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.base = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
        )
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()

        hidden_size = int(self.base.config.hidden_size)
        self.start_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.end_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, 2)
        nn.init.xavier_uniform_(self.start_query.weight)
        nn.init.xavier_uniform_(self.end_query.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        # fixed scaling keeps the initial pointer logits of a large LM in a sane range
        self.score_scale = float(score_scale) if score_scale is not None else 1.0 / math.sqrt(hidden_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor, past_key_values=None, use_cache: bool = False):
        decoder = getattr(self.base, self.base.base_model_prefix)
        output = decoder(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=False,
            return_dict=True,
        )
        return output.last_hidden_state, output.past_key_values

    @torch.no_grad()
    def encode_source(self, source_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.hidden_states(source_ids)
        return hidden[0].float()

    @torch.no_grad()
    def vocab_logits(self, decoder_state: torch.Tensor) -> torch.Tensor:
        lm_head = self.base.get_output_embeddings()
        return lm_head(decoder_state.to(lm_head.weight.dtype)).float()

    def endpoint_scores(self, decoder_states: torch.Tensor, source_states: torch.Tensor):
        decoder_states = decoder_states.float()
        source_transposed = source_states.float().transpose(0, 1)
        start_scores = torch.matmul(self.start_query(decoder_states), source_transposed) * self.score_scale
        end_scores = torch.matmul(self.end_query(decoder_states), source_transposed) * self.score_scale
        return start_scores, end_scores

    def gate_logits(self, decoder_states: torch.Tensor) -> torch.Tensor:
        return self.gate(decoder_states.float()).float()

    def save_heads(self, path: Path, config: dict) -> None:
        heads = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.split(".")[0] in ("start_query", "end_query", "gate")
        }
        checkpoint = {
            "model_path": self.model_path,
            "hidden_size": int(self.base.config.hidden_size),
            "score_scale": self.score_scale,
            "head_state_dict": heads,
            "config": config,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_heads(self, checkpoint: dict) -> None:
        if int(checkpoint["hidden_size"]) != int(self.base.config.hidden_size):
            raise ValueError("checkpoint was trained for a different base model")
        self.score_scale = float(checkpoint["score_scale"])
        self.load_state_dict(checkpoint["head_state_dict"], strict=False)
