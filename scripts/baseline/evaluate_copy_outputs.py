#!/usr/bin/env python3
"""Exact-match scoring of copy-task predictions against an oracle JSONL."""

from __future__ import annotations

import argparse
import io
import json
import re
import tokenize
from pathlib import Path


CODE_BLOCK_TARGET = "first_python_or_fenced_code_block"
FENCE_RE = re.compile(r"(?ms)^[ \t]*`{3,}(?:python)?[ \t]*\n(.*?)^[ \t]*`{3,}[ \t]*$")


def get_prediction(row: dict) -> str:
    for key in ("prediction", "model_edited_answer"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def _line_start(text: str, offset: int) -> int:
    return text.rfind("\n", 0, offset) + 1


def _include_closing_line_comma(text: str, offset: int) -> int:
    line_end = text.find("\n", offset)
    if line_end < 0:
        line_end = len(text)
    suffix = text[offset:line_end]
    match = re.match(r"[ \t]*,?", suffix)
    return offset + (match.end() if match else 0)


def _extract_bracketed_block(
    text: str,
    *,
    line_start: int,
    opening_offset: int,
) -> str | None:
    pairs = {")": "(", "]": "[", "}": "{"}
    opening_chars = set(pairs.values())
    stack: list[str] = []
    quote: str | None = None
    triple_quote = False
    escaped = False
    in_comment = False
    offset = opening_offset

    while offset < len(text):
        char = text[offset]
        if in_comment:
            if char == "\n":
                in_comment = False
            offset += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
                offset += 1
                continue
            if char == "\\":
                escaped = True
                offset += 1
                continue
            if triple_quote and text.startswith(quote * 3, offset):
                quote = None
                triple_quote = False
                offset += 3
                continue
            if not triple_quote and char == quote:
                quote = None
            offset += 1
            continue
        if char == "#":
            in_comment = True
            offset += 1
            continue
        if char in {'"', "'"}:
            quote = char
            triple_quote = text.startswith(char * 3, offset)
            offset += 3 if triple_quote else 1
            continue
        if char in opening_chars:
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return None
            stack.pop()
            if not stack:
                end = _include_closing_line_comma(text, offset + 1)
                return text[line_start:end]
        offset += 1
    return None


def _extract_anchored_block_at(text: str, anchor: str, anchor_offset: int) -> str | None:
    start = _line_start(text, anchor_offset)
    if text[start:anchor_offset].strip():
        return None
    anchor_line_end = text.find("\n", anchor_offset)
    if anchor_line_end < 0:
        anchor_line_end = len(text)
    anchor_line = text[start:anchor_line_end]
    anchor_column = anchor_offset - start
    search_end = anchor_column + len(anchor)
    opening_column = max(anchor_line.rfind(char, anchor_column, search_end + 1) for char in "([{")
    if opening_column < 0:
        return None
    return _extract_bracketed_block(text, line_start=start, opening_offset=start + opening_column)


def _fragment_tokens(text: str) -> list[tokenize.TokenInfo]:
    """Tokenize an incomplete Python line, keeping the tokens emitted before EOF."""
    ignored_types = {
        tokenize.ENCODING,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.ENDMARKER,
    }
    tokens: list[tokenize.TokenInfo] = []
    generator = tokenize.generate_tokens(io.StringIO(text).readline)
    try:
        for token in generator:
            if token.type in ignored_types:
                continue
            if token.type == tokenize.ERRORTOKEN and token.string.isspace():
                continue
            tokens.append(token)
    except (IndentationError, SyntaxError, tokenize.TokenError):
        pass
    return tokens


def extract_token_anchored_python_block(text: str, anchor: str) -> str | None:
    """Find the anchor line while allowing whitespace-only changes in token layout."""
    anchor_tokens = _fragment_tokens(anchor)
    anchor_signature = [(token.type, token.string) for token in anchor_tokens]
    opening_indices = [
        index for index, token in enumerate(anchor_tokens) if token.type == tokenize.OP and token.string in "([{"
    ]
    if not anchor_signature or not opening_indices:
        return None
    opening_index = opening_indices[-1]

    line_start = 0
    for line in text.splitlines(keepends=True):
        candidate_tokens = _fragment_tokens(line)
        candidate_signature = [(token.type, token.string) for token in candidate_tokens]
        if candidate_signature[: len(anchor_signature)] == anchor_signature:
            opening_column = candidate_tokens[opening_index].start[1]
            extracted = _extract_bracketed_block(
                text,
                line_start=line_start,
                opening_offset=line_start + opening_column,
            )
            if extracted is not None:
                return extracted
        line_start += len(line)
    return None


def extract_anchored_python_block(text: str, oracle_row: dict) -> str | None:
    """Extract the first bracketed block that starts at the Code task's anchor line."""
    anchor = oracle_row.get("metadata", {}).get("anchor_text")
    if not isinstance(anchor, str) or not anchor.strip():
        return None
    anchor = anchor.strip()

    search_from = 0
    while True:
        anchor_offset = text.find(anchor, search_from)
        if anchor_offset < 0:
            return None
        extracted = _extract_anchored_block_at(text, anchor, anchor_offset)
        if extracted is not None:
            return extracted
        search_from = anchor_offset + max(len(anchor), 1)


def extract_first_fenced_block(text: str) -> str | None:
    match = FENCE_RE.search(text)
    return match.group(1) if match else None


def task_target_and_prediction(oracle_row: dict, prediction: str) -> tuple[str, str, str, bool]:
    """Return (evaluation target, gold text, prediction text to compare, extraction succeeded)."""
    metadata = oracle_row.get("metadata", {})
    evaluation_target = metadata.get("evaluation_target", "full_output")
    if evaluation_target != CODE_BLOCK_TARGET:
        gold = oracle_row.get("gold_answer", oracle_row.get("edit_content", ""))
        return str(evaluation_target), str(gold), prediction, True

    gold = oracle_row.get("gold_evidence", "")
    extracted = extract_anchored_python_block(prediction, oracle_row)
    if extracted is None:
        anchor = metadata.get("anchor_text")
        if isinstance(anchor, str):
            extracted = extract_token_anchored_python_block(prediction, anchor.strip())
    if extracted is None:
        extracted = extract_first_fenced_block(prediction)
    return CODE_BLOCK_TARGET, str(gold), extracted or "", extracted is not None


def _nonempty_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [line.rstrip() for line in normalized.split("\n") if line.strip()]


def normalize_task_text(text: str) -> str:
    """Normalize newlines, strip trailing whitespace and drop empty lines."""
    return "\n".join(_nonempty_lines(text)).strip()


def _code_tokens(text: str) -> tuple[tuple[int, str], ...]:
    """Python tokens with a uniform outer indent removed; relative indentation is kept."""
    ignored_types = {tokenize.ENCODING, tokenize.NL, tokenize.ENDMARKER}
    try:
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO("\n".join(_nonempty_lines(text))).readline)
            if token.type not in ignored_types
        ]
        if tokens and tokens[0].type == tokenize.INDENT:
            tokens.pop(0)
            for index in range(len(tokens) - 1, -1, -1):
                if tokens[index].type == tokenize.DEDENT:
                    tokens.pop(index)
                    break
        layout_types = {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE}
        return tuple((token.type, "" if token.type in layout_types else token.string) for token in tokens)
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return ((tokenize.ERRORTOKEN, normalize_task_text(text)),)


def _drop_terminal_comma(text: str) -> str:
    terminal_offset = len(text.rstrip())
    if terminal_offset and text[terminal_offset - 1] == ",":
        text = text[: terminal_offset - 1] + text[terminal_offset:]
    return text


def task_text_matches(evaluation_target: str, prediction: str, gold: str) -> bool:
    """Code blocks are compared as Python tokens, ignoring a trailing comma."""
    if evaluation_target == CODE_BLOCK_TARGET:
        return _code_tokens(_drop_terminal_comma(prediction)) == _code_tokens(_drop_terminal_comma(gold))
    return normalize_task_text(prediction) == normalize_task_text(gold)


def unwrap_tagged_output(prediction: str, tag: str) -> str | None:
    """Return the text between a first line <tag> and a last line </tag>, or None if the format is not met."""
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    if prediction.count(open_tag) != 1 or prediction.count(close_tag) != 1:
        return None
    if not prediction.startswith(open_tag + "\n") or not prediction.endswith("\n" + close_tag):
        return None
    return prediction[len(open_tag) + 1 : -len(close_tag) - 1]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rows_by_task_id(rows: list[dict], *, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id in indexed:
            raise SystemExit(f"{label} contains duplicate task_id={task_id!r}")
        indexed[task_id] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate copy-task outputs with exact match.")
    parser.add_argument("--input", required=True, help="Predictions JSONL path")
    parser.add_argument("--oracle", required=True, help="Oracle JSONL with the gold answers and task metadata")
    parser.add_argument("--summary-out", default="", help="Optional summary JSON output")
    parser.add_argument(
        "--output-tag",
        default="",
        help="If set, a prediction must be wrapped in <TAG>...</TAG> lines; only the wrapped text is scored.",
    )
    args = parser.parse_args()

    oracle_rows = load_jsonl(Path(args.oracle))
    predictions_by_id = rows_by_task_id(load_jsonl(Path(args.input)), label="predictions")
    oracle_ids = rows_by_task_id(oracle_rows, label="oracle").keys()
    if predictions_by_id.keys() != oracle_ids:
        missing = sorted(oracle_ids - predictions_by_id.keys())
        unexpected = sorted(predictions_by_id.keys() - oracle_ids)
        raise SystemExit(
            f"predictions do not cover the oracle: missing={len(missing)} {missing[:3]}, "
            f"unexpected={len(unexpected)} {unexpected[:3]}"
        )

    total = len(oracle_rows)
    exact = 0
    format_valid = 0
    for oracle_row in oracle_rows:
        pred = get_prediction(predictions_by_id[oracle_row["task_id"]])
        if args.output_tag:
            pred = unwrap_tagged_output(pred, args.output_tag)
            if pred is None:
                continue
            format_valid += 1
        evaluation_target, gold, task_pred, _ = task_target_and_prediction(oracle_row, pred)
        exact += int(task_text_matches(evaluation_target, task_pred, gold))

    exact_rate = exact / total if total else 0.0
    summary = {
        "input": args.input,
        "oracle": args.oracle,
        "total_rows": total,
        "exact_match_count": exact,
        "exact_match_rate": exact_rate,
        "exact_match_percent": round(exact_rate * 100, 2),
    }
    if args.output_tag:
        summary["format_valid_count"] = format_valid

    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"rows={total}")
    print(f"exact_match_rate={summary['exact_match_percent']:.2f}%")
    if args.output_tag:
        print(f"format_valid={format_valid}/{total}")
    if args.summary_out:
        print(f"summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
