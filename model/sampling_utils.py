"""Temperature, top-p and min-p token selection shared by the decoders."""

from __future__ import annotations

import torch
from transformers import MinPLogitsWarper, TemperatureLogitsWarper, TopPLogitsWarper


def validate_sampling_args(temperature: float, top_p: float, min_p: float) -> None:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not 0 <= min_p <= 1:
        raise ValueError("min_p must be in [0, 1]")


def filter_sampling_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    min_p: float,
) -> torch.Tensor:
    """Apply temperature, top-p and min-p filtering to one logit vector."""
    validate_sampling_args(temperature, top_p, min_p)
    if temperature == 0:
        return logits

    # Warpers run in generation order; logits keep their input dtype.
    input_ids = torch.empty((1, 0), dtype=torch.long, device=logits.device)
    filtered = logits.unsqueeze(0)
    if temperature != 1:
        filtered = TemperatureLogitsWarper(temperature)(input_ids, filtered)
    if top_p < 1:
        filtered = TopPLogitsWarper(top_p)(input_ids, filtered)
    if min_p > 0:
        filtered = MinPLogitsWarper(min_p)(input_ids, filtered)
    return filtered[0]


def select_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    generator: torch.Generator | None = None,
) -> int:
    """Greedy at temperature 0, otherwise sample from the filtered distribution."""
    if temperature == 0:
        return int(torch.argmax(logits).item())
    filtered = filter_sampling_logits(logits, temperature=temperature, top_p=top_p, min_p=min_p)
    probs = torch.softmax(filtered, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
