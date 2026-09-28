#!/usr/bin/env python3
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TRIGGER_TO_ID = {"GEN": 0, "COPY": 1}


class CopyConstraintModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        freeze_base: bool = True,
        torch_dtype: torch.dtype | None = None,
        dropout: float = 0.1,
        source_bias_scale: float = 6.0,
    ):
        super().__init__()
        tokenizer_path = adapter_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch_dtype,
            )
        except TypeError:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
        if adapter_path:
            base_model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=not freeze_base,
            )
        self.base = base_model
        self.config = self.base.config
        hidden_size = self.config.hidden_size
        plugin_input_dim = hidden_size + 1

        self.plugin_input_norm = nn.LayerNorm(plugin_input_dim)
        self.trigger_head = nn.Sequential(
            nn.Linear(plugin_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )
        self.length_head = nn.Sequential(
            nn.Linear(plugin_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.source_bias_scale = float(source_bias_scale)
        self.source_bias_input_norm = nn.LayerNorm(hidden_size + 1)
        self.source_bias_head = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.source_bias_head[-1].weight)
        nn.init.zeros_(self.source_bias_head[-1].bias)
        self._align_plugin_dtype_with_base()

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False
            self.base.eval()

    @property
    def base_is_frozen(self) -> bool:
        return not any(param.requires_grad for param in self.base.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.base_is_frozen:
            self.base.eval()
        return self

    def _base_forward_context(self):
        return torch.no_grad() if self.base_is_frozen else nullcontext()

    def _base_parameter_dtype(self) -> torch.dtype:
        try:
            return next(self.base.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _align_plugin_dtype_with_base(self) -> None:
        target_dtype = self._base_parameter_dtype()
        self.plugin_input_norm.to(dtype=target_dtype)
        self.trigger_head.to(dtype=target_dtype)
        self.length_head.to(dtype=target_dtype)
        self.source_bias_input_norm.to(dtype=target_dtype)
        self.source_bias_head.to(dtype=target_dtype)

    def _causal_lm(self) -> nn.Module:
        if isinstance(self.base, PeftModel):
            return self.base.get_base_model()
        return self.base

    def _decoder_backbone(self) -> nn.Module:
        causal_lm = self._causal_lm()
        base_model_prefix = getattr(causal_lm, "base_model_prefix", "")
        if base_model_prefix and hasattr(causal_lm, base_model_prefix):
            return getattr(causal_lm, base_model_prefix)
        return getattr(causal_lm, "model", causal_lm)

    @property
    def lm_head(self) -> nn.Module:
        return self._causal_lm().lm_head

    def decode_hidden_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        decoder = self._decoder_backbone()
        with self._base_forward_context():
            out = decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=False,
                return_dict=True,
            )
        return out.last_hidden_state, out.past_key_values

    def decode_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        hidden_states, past_key_values = self.decode_hidden_sequence(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return hidden_states[:, -1, :], past_key_values

    def vocab_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        return self.lm_head(hidden_states)

    def proposal_predictor_states_and_logits(
        self,
        hidden_state: torch.Tensor,
        past_key_values,
        proposal_ids: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not proposal_ids:
            hidden_size = self.config.hidden_size
            empty_hidden = torch.empty((0, hidden_size), dtype=self._base_parameter_dtype(), device=device)
            empty_logits = torch.empty((0, self.config.vocab_size), dtype=torch.float32, device=device)
            return empty_hidden, empty_logits

        first_hidden = hidden_state
        if first_hidden.ndim == 1:
            first_hidden = first_hidden.unsqueeze(0)

        if len(proposal_ids) > 1:
            consumed = torch.tensor([proposal_ids[:-1]], dtype=torch.long, device=device)
            verify_past = past_key_values
            restore_cache_len: int | None = None
            if past_key_values is not None:
                if hasattr(past_key_values, "get_seq_length") and hasattr(past_key_values, "crop"):
                    restore_cache_len = int(past_key_values.get_seq_length())
                else:
                    # Some cache implementations mutate in-place even when use_cache=False.
                    verify_past = copy.deepcopy(past_key_values)
            try:
                tail_hidden, _ = self.decode_hidden_sequence(
                    consumed,
                    past_key_values=verify_past,
                    use_cache=False,
                )
            finally:
                if restore_cache_len is not None:
                    past_key_values.crop(restore_cache_len)
            predictor_hidden = torch.cat([first_hidden.unsqueeze(1), tail_hidden], dim=1)[0]
        else:
            predictor_hidden = first_hidden

        predictor_hidden = predictor_hidden.squeeze(0) if predictor_hidden.ndim == 3 else predictor_hidden
        logits = self.vocab_logits(predictor_hidden).float()
        return predictor_hidden, logits

    def _plugin_inputs(
        self,
        hidden_states: torch.Tensor,
        match_lengths,
    ) -> torch.Tensor:
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(0)

        device = hidden_states.device
        dtype = self.plugin_input_norm.weight.dtype

        if not torch.is_tensor(match_lengths):
            match_lengths = torch.tensor(match_lengths, device=device, dtype=dtype)
        else:
            match_lengths = match_lengths.to(device=device, dtype=dtype)

        if match_lengths.ndim == 0:
            match_lengths = match_lengths.unsqueeze(0)

        match_feat = torch.log1p(match_lengths.clamp_min(0.0)).unsqueeze(-1)

        hidden_states = hidden_states.to(dtype)
        plugin_inputs = torch.cat([hidden_states, match_feat], dim=-1)
        return self.plugin_input_norm(plugin_inputs)

    def trigger_logits(
        self,
        hidden_states: torch.Tensor,
        match_lengths,
    ) -> torch.Tensor:
        return self.trigger_head(self._plugin_inputs(hidden_states, match_lengths))

    def length_log(
        self,
        hidden_states: torch.Tensor,
        match_lengths,
    ) -> torch.Tensor:
        raw = self.length_head(self._plugin_inputs(hidden_states, match_lengths))
        return F.softplus(raw)

    # Backward-compatible alias for older training scripts.
    length_log1p = length_log

    def predict_copy_length(
        self,
        hidden_states: torch.Tensor,
        match_lengths,
    ) -> torch.Tensor:
        # COPY mode proposes a positive span length.
        log_lengths = self.length_log(hidden_states, match_lengths)
        return torch.floor(torch.exp(log_lengths)).long().clamp_min(1)

    def _source_bias_inputs(
        self,
        hidden_states: torch.Tensor,
        raw_margins,
    ) -> torch.Tensor:
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(0)

        device = hidden_states.device
        dtype = self.source_bias_input_norm.weight.dtype

        if not torch.is_tensor(raw_margins):
            raw_margins = torch.tensor(raw_margins, device=device, dtype=dtype)
        else:
            raw_margins = raw_margins.to(device=device, dtype=dtype)

        if raw_margins.ndim == 0:
            raw_margins = raw_margins.unsqueeze(0)

        hidden_states = hidden_states.to(dtype)
        margin_feat = raw_margins.unsqueeze(-1)
        inputs = torch.cat([hidden_states, margin_feat], dim=-1)
        return self.source_bias_input_norm(inputs)

    def source_token_bias(
        self,
        hidden_states: torch.Tensor,
        raw_margins,
    ) -> torch.Tensor:
        raw = self.source_bias_head(self._source_bias_inputs(hidden_states, raw_margins)).squeeze(-1)
        return self.source_bias_scale * torch.tanh(raw)

    def plugin_state_dict(self) -> dict[str, dict]:
        return {
            "plugin_input_norm": self.plugin_input_norm.state_dict(),
            "trigger_head": self.trigger_head.state_dict(),
            "length_head": self.length_head.state_dict(),
            "source_bias_input_norm": self.source_bias_input_norm.state_dict(),
            "source_bias_head": self.source_bias_head.state_dict(),
        }

    def load_plugin_state_dict(self, state: dict[str, dict]) -> None:
        required = {"plugin_input_norm", "trigger_head", "length_head"}
        missing = sorted(required.difference(state))
        if missing:
            raise KeyError(f"Missing plugin state keys: {', '.join(missing)}")
        self.plugin_input_norm.load_state_dict(state["plugin_input_norm"])
        self.trigger_head.load_state_dict(state["trigger_head"])
        self.length_head.load_state_dict(state["length_head"])
        if "source_bias_input_norm" in state:
            self.source_bias_input_norm.load_state_dict(state["source_bias_input_norm"])
        if "source_bias_head" in state:
            self.source_bias_head.load_state_dict(state["source_bias_head"])


# Backward-compatible alias for older imports and checkpoints.
QwenCopyConstraintModel = CopyConstraintModel


def trigger_nll_loss(trigger_logits: torch.Tensor, target_trigger_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(trigger_logits, target_trigger_ids)


def asymmetric_length_regression_loss(
    pred_log_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    over_weight: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    # Kept under the old name for checkpoint / import compatibility.
    # The current objective is symmetric Huber regression over log length.
    pred = pred_log_lengths.squeeze(-1).float()
    target = torch.log(target_lengths.float().clamp_min(1.0))
    diff = pred - target
    abs_diff = diff.abs()

    if beta > 0:
        huber = torch.where(abs_diff <= beta, 0.5 * diff.square() / beta, abs_diff - 0.5 * beta)
    else:
        huber = abs_diff

    return huber.mean()


def source_bias_margin_hinge_loss(
    raw_margins: torch.Tensor,
    bias: torch.Tensor,
    accept_labels: torch.Tensor,
    pos_margin: float = 1.0,
    neg_margin: float = 1.0,
) -> torch.Tensor:
    adjusted_margins = raw_margins.float() + bias.float()
    labels = accept_labels.float()

    pos_loss = torch.clamp(pos_margin - adjusted_margins, min=0.0)
    neg_loss = torch.clamp(adjusted_margins + neg_margin, min=0.0)
    return torch.where(labels > 0.5, pos_loss, neg_loss).mean()
