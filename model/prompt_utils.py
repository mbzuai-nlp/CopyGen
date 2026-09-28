#!/usr/bin/env python3
from __future__ import annotations

import torch
from typing import Optional


def build_prompt_messages(context: str, instruction: str) -> list[dict[str, str]]:
    system = (
        "You are a precise long-context text assistant. "
        "Follow the user's instruction over the provided source text. "
        "When the instruction asks for exact copying or quoting, preserve the source wording exactly. "
        "Output only the requested final answer. "
        "Do not add explanations, notes, markdown fences, or any extra content."
    )
    user = (
        "Instruction:\n\n"
        f"{instruction}\n\n"
        "Source text:\n\n"
        f"{context}\n\n"
        "Constraint:\n"
        "Return only the requested answer. Nothing else."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _render_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    disable_thinking: Optional[bool] = None,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    chat_template = getattr(tokenizer, "chat_template", None)
    supports_enable_thinking = isinstance(chat_template, str) and "enable_thinking" in chat_template
    should_disable_thinking = supports_enable_thinking if disable_thinking is None else (disable_thinking and supports_enable_thinking)
    if should_disable_thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def render_editor_prompt(
    tokenizer,
    context: str,
    instruction: str,
    disable_thinking: Optional[bool] = None,
) -> str:
    messages = build_prompt_messages(context, instruction)
    if hasattr(tokenizer, "apply_chat_template"):
        return _render_chat_template(
            tokenizer,
            messages,
            add_generation_prompt=True,
            disable_thinking=disable_thinking,
        )

    return (
        "System:\n"
        f"{messages[0]['content']}\n\n"
        "User:\n"
        f"{messages[1]['content']}\n\n"
        "Assistant:\n"
    )


def encode_editor_prefix(
    tokenizer,
    context: str,
    instruction: str,
    device: torch.device,
    disable_thinking: Optional[bool] = None,
) -> torch.Tensor:
    rendered = render_editor_prompt(
        tokenizer,
        context,
        instruction,
        disable_thinking=disable_thinking,
    )
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids], dtype=torch.long, device=device)


def build_source_messages(context: str, instruction: str) -> list[dict[str, str]]:
    system = (
        "You are a precise long-context text assistant. "
        "Follow the user's instruction over the provided source text."
    )
    user = (
        "Instruction:\n\n"
        f"{instruction}\n\n"
        "Source text:\n\n"
        f"{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_source_prompt(
    tokenizer,
    context: str,
    instruction: str,
    disable_thinking: Optional[bool] = None,
) -> str:
    messages = build_source_messages(context, instruction)
    if hasattr(tokenizer, "apply_chat_template"):
        return _render_chat_template(
            tokenizer,
            messages,
            add_generation_prompt=False,
            disable_thinking=disable_thinking,
        )

    return (
        "System:\n"
        f"{messages[0]['content']}\n\n"
        "User:\n"
        f"{messages[1]['content']}"
    )


def encode_instruction_aware_source(
    tokenizer,
    context: str,
    instruction: str,
    device: torch.device,
    disable_thinking: Optional[bool] = None,
):
    """Encode a full chat-formatted source prompt and return the context token slice."""
    rendered = render_source_prompt(
        tokenizer,
        context,
        instruction,
        disable_thinking=disable_thinking,
    )
    source_prefix = (
        "Instruction:\n\n"
        f"{instruction}\n\n"
        "Source text:\n\n"
    )
    user_content = source_prefix + context
    user_start = rendered.find(user_content)
    if user_start < 0:
        raise ValueError("Could not locate source user content inside rendered prompt")

    ctx_char_start = user_start + len(source_prefix)
    ctx_char_end = ctx_char_start + len(context)

    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        ctx_token_idxs = [
            idx
            for idx, (start, end) in enumerate(offsets)
            if end > ctx_char_start and start < ctx_char_end
        ]
        if not ctx_token_idxs:
            raise ValueError("Could not locate context token span inside rendered prompt")
        if ctx_token_idxs != list(range(ctx_token_idxs[0], ctx_token_idxs[-1] + 1)):
            raise ValueError("Context token span is unexpectedly non-contiguous")
        ctx_start = ctx_token_idxs[0]
        ctx_end = ctx_token_idxs[-1] + 1
        context_ids = full_ids[ctx_start:ctx_end]
    else:
        raise ValueError("encode_instruction_aware_source requires a fast tokenizer")

    return (
        torch.tensor([full_ids], dtype=torch.long, device=device),
        slice(ctx_start, ctx_end),
        context_ids,
    )
