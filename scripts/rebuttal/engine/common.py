"""CopyGen heads and small data helpers shared by the vLLM and SGLang integrations."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_leaf_module(name: str, relative_path: str):
    # model/__init__.py imports training dependencies (peft) that are not
    # installed in the engine environments, so load the leaf modules directly.
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SourceMatcher = _load_leaf_module("copygen_matcher", "model/matcher.py").SourceMatcher
_prompt_utils = _load_leaf_module("copygen_prompt_utils", "model/prompt_utils.py")
render_editor_prompt = _prompt_utils.render_editor_prompt


class CopyGenHeads(nn.Module):
    """Trigger, length and source-bias heads of a trained CopyGen plugin."""

    def __init__(self, hidden_size: int, source_bias_scale: float) -> None:
        super().__init__()
        input_size = hidden_size + 1
        self.plugin_input_norm = nn.LayerNorm(input_size)
        self.trigger_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_size, 2),
        )
        self.length_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_size, 1),
        )
        self.source_bias_scale = float(source_bias_scale)
        self.source_bias_input_norm = nn.LayerNorm(input_size)
        self.source_bias_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_size, 1),
        )

    def load_plugin_state(self, state: dict) -> None:
        self.plugin_input_norm.load_state_dict(state["plugin_input_norm"])
        self.trigger_head.load_state_dict(state["trigger_head"])
        self.length_head.load_state_dict(state["length_head"])
        self.source_bias_input_norm.load_state_dict(state["source_bias_input_norm"])
        self.source_bias_head.load_state_dict(state["source_bias_head"])

    def _plugin_inputs(self, hidden_states, match_lengths):
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(0)
        dtype = self.plugin_input_norm.weight.dtype
        match = torch.as_tensor(match_lengths, dtype=dtype, device=hidden_states.device)
        if match.ndim == 0:
            match = match.unsqueeze(0)
        match_feature = torch.log1p(match.clamp_min(0.0)).unsqueeze(-1)
        inputs = torch.cat([hidden_states.to(dtype), match_feature], dim=-1)
        return self.plugin_input_norm(inputs)

    def trigger_logits(self, hidden_states, match_lengths):
        return self.trigger_head(self._plugin_inputs(hidden_states, match_lengths))

    def length_log(self, hidden_states, match_lengths):
        return F.softplus(self.length_head(self._plugin_inputs(hidden_states, match_lengths)))

    def predict_copy_length(self, hidden_states, match_lengths):
        length_log = self.length_log(hidden_states, match_lengths)
        return torch.floor(torch.exp(length_log)).long().clamp_min(1)

    def source_token_bias(self, hidden_states, raw_margins):
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(0)
        dtype = self.source_bias_input_norm.weight.dtype
        hidden_states = hidden_states.to(dtype)
        raw_margins = raw_margins.to(device=hidden_states.device, dtype=dtype).reshape(-1)
        inputs = torch.cat([hidden_states, raw_margins.unsqueeze(-1)], dim=-1)
        raw = self.source_bias_head(self.source_bias_input_norm(inputs)).squeeze(-1)
        return self.source_bias_scale * torch.tanh(raw)


def run_heads_on_zeros(heads, mode: str, hidden, matches, bias_hidden, margins):
    """Execute the heads with the shapes of one decode step and return a sink.

    ``bias`` runs the source-bias head on every candidate item; ``all3`` also
    runs the trigger and length heads. The caller multiplies the sink by zero.
    """

    def plugin_inputs():
        match_feature = torch.log1p(matches.clamp_min(0.0)).unsqueeze(-1)
        return heads.plugin_input_norm(torch.cat([hidden, match_feature], dim=-1))

    request_count = hidden.shape[0]
    sink = torch.zeros((request_count,), dtype=hidden.dtype, device=hidden.device)
    if mode == "all3":
        sink = sink + heads.trigger_head(plugin_inputs()).sum(dim=-1)
        sink = sink + F.softplus(heads.length_head(plugin_inputs())).squeeze(-1)
    bias_inputs = torch.cat([bias_hidden, margins.unsqueeze(-1)], dim=-1)
    raw = heads.source_bias_head(heads.source_bias_input_norm(bias_inputs)).squeeze(-1)
    bias = heads.source_bias_scale * torch.tanh(raw)
    return sink + bias.view(request_count, -1).sum(dim=-1)


def load_heads(plugin_path, device, dtype=torch.bfloat16) -> CopyGenHeads:
    checkpoint = torch.load(plugin_path, map_location="cpu", weights_only=True)
    state = checkpoint["plugin_state_dict"]
    hidden_size = state["plugin_input_norm"]["weight"].numel() - 1
    heads = CopyGenHeads(hidden_size, float(checkpoint.get("source_bias_scale", 6.0)))
    heads.load_plugin_state(state)
    return heads.to(device=device, dtype=dtype).eval()


def load_jsonl(path, limit: int = 0) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_domain_paths(values: list[str]) -> dict[str, str]:
    """Parse repeated DOMAIN=PATH arguments, keeping their order."""
    parsed = {}
    for value in values:
        domain, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"expected DOMAIN=PATH, got {value!r}")
        parsed[domain] = path
    return parsed
