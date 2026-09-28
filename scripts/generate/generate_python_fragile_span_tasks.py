#!/usr/bin/env python3
"""Generate anchored fragile Python span-copy tasks.

The target is not a whole function. Each task points to a clear anchor inside a
function, then asks for the exact nested dict/list/argument block at that anchor.
This keeps localization mostly mechanical while making exact copying difficult.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import keyword
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")


@dataclass
class SpanBlock:
    repo: str
    relative_path: str
    function_name: str
    qualname: str
    function_start_line: int
    function_end_line: int
    function_source: str
    context: str
    context_start_line: int
    context_end_line: int
    target_source: str
    target_start_line: int
    target_end_line: int
    target_lines: int
    block_type: str
    anchor_text: str
    instruction_anchor: str
    explanation: str
    fragile_score: int
    features: dict[str, int | str]


@dataclass
class Task:
    task_id: str
    case_id: str
    task_type: str
    span_type: str
    source_split: str
    source_dataset: str
    instruction: str
    context: str
    gold_answer: str
    gold_evidence: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/code"))
    parser.add_argument(
        "--repo-allowlist",
        default=(
            "fastapi-fastapi,psf-requests,encode-starlette,fastapi-sqlmodel,"
            "fastapi-users-fastapi-users,ansible-documentation,kubernetes-website"
        ),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/code/train.jsonl"),
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("data/code/test.jsonl"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("data/code/stats.json"),
    )
    parser.add_argument("--min-function-lines", type=int, default=20)
    parser.add_argument("--max-function-lines", type=int, default=260)
    parser.add_argument("--min-target-lines", type=int, default=18)
    parser.add_argument("--max-target-lines", type=int, default=180)
    parser.add_argument("--min-context-lines", type=int, default=120)
    parser.add_argument("--max-context-lines", type=int, default=1000)
    parser.add_argument("--max-context-chars", type=int, default=50000)
    parser.add_argument("--target-total", type=int, default=1050)
    parser.add_argument("--val-count", type=int, default=229)
    parser.add_argument("--train-variants-per-block", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def repo_name(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root).parts[0]


def is_allowed_file(path: Path) -> bool:
    blocked = {".git", ".venv", "venv", "__pycache__", "site-packages"}
    return not any(part in blocked for part in path.parts)


def split_words(name: str) -> list[str]:
    parts: list[str] = []
    for chunk in re.split(r"[_\W]+", name):
        if not chunk:
            continue
        parts.extend(m.group(0).lower() for m in WORD_RE.finditer(chunk))
    return [part for part in parts if part and not keyword.iskeyword(part)]


def function_source(lines: list[str], node: ast.AST) -> tuple[str, int, int]:
    start = int(getattr(node, "lineno", 1))
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        start = min(start, *(int(getattr(dec, "lineno", start)) for dec in decorators))
    end = int(getattr(node, "end_lineno", start))
    return "\n".join(lines[start - 1 : end]) + "\n", start, end


def iter_function_nodes(tree: ast.AST) -> list[tuple[list[str], ast.AST]]:
    result: list[tuple[list[str], ast.AST]] = []

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = parents + [child.name]
                result.append((qual, child))
                visit(child, qual)
            elif isinstance(child, ast.ClassDef):
                visit(child, parents + [child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return result


def build_context(
    lines: list[str],
    function_start: int,
    function_end: int,
    max_lines: int,
    min_lines: int,
    max_chars: int,
) -> tuple[str, int, int]:
    total = len(lines)
    fn_len = function_end - function_start + 1
    window = min(max_lines, max(min_lines, fn_len + 120))
    before = max(0, (window - fn_len) // 2)
    ctx_start = max(1, function_start - before)
    ctx_end = min(total, ctx_start + window - 1)
    ctx_start = max(1, min(ctx_start, max(1, ctx_end - window + 1)))

    while ctx_end >= ctx_start:
        text = "\n".join(lines[ctx_start - 1 : ctx_end]) + "\n"
        if len(text) <= max_chars or ctx_end - ctx_start + 1 <= fn_len:
            return text, ctx_start, ctx_end
        if function_start - ctx_start > ctx_end - function_end:
            ctx_start += 10
        else:
            ctx_end -= 10
    return "\n".join(lines[function_start - 1 : function_end]) + "\n", function_start, function_end


def source_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1 : end]) + "\n"


def clean_anchor(line: str) -> str:
    return line.strip()


def is_string_constant(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def key_to_text(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant):
        return repr(node.value)
    try:
        return ast.unparse(node) if node is not None else "<unknown>"
    except Exception:
        return "<unknown>"


def block_features(source: str, target_lines: int) -> dict[str, int | str]:
    return {
        "target_lines": target_lines,
        "string_literals_approx": len(re.findall(r"(?<!\\)['\"]", source)) // 2,
        "brace_count": sum(source.count(ch) for ch in "{}[]()"),
        "punctuation_count": sum(source.count(ch) for ch in "{}[]():,'\""),
        "comma_count": source.count(","),
        "quote_count": source.count('"') + source.count("'"),
        "path_like_count": len(re.findall(r"/[A-Za-z0-9_./{}:-]+", source)),
        "max_indent": max(
            ((len(line) - len(line.lstrip(" "))) // 4 for line in source.splitlines() if line.strip()),
            default=0,
        ),
    }


def fragile_score(source: str, target_lines: int) -> int:
    features = block_features(source, target_lines)
    return (
        target_lines * 8
        + int(features["brace_count"]) * 3
        + int(features["punctuation_count"])
        + int(features["quote_count"]) * 2
        + int(features["path_like_count"]) * 12
        + int(features["max_indent"]) * 8
    )


def anchor_key(anchor: str) -> str:
    match = re.match(r"""['"]([^'"]+)['"]\s*:""", anchor.strip())
    return match.group(1) if match else anchor.strip().split(":", 1)[0].strip()


def make_explanation(block_type: str, function_name: str, anchor: str) -> str:
    words = " ".join(split_words(function_name)) or function_name
    key = anchor_key(anchor)
    if block_type == "key_value_block":
        key_descriptions = {
            "paths": "This block lists the expected OpenAPI path entries that the schema test compares.",
            "components": "This block contains the reusable OpenAPI components expected in the schema output.",
            "schemas": "This block defines the expected component schemas and their nested fields.",
            "properties": "This block records the expected field properties for the surrounding schema object.",
            "responses": "This block specifies the expected response metadata and response schemas for a route.",
            "requestBody": "This block specifies the expected request body schema for the surrounding operation.",
            "content": "This block records the expected media-type content mapping for a response or request body.",
            "parameters": "This block lists the expected OpenAPI parameters for the surrounding operation.",
            "callbacks": "This block records the expected callback operation schema for the endpoint.",
            "required": "This block lists the required fields expected by the surrounding schema.",
            "anyOf": "This block lists the expected alternative schema shapes for the surrounding field.",
            "items": "This block defines the expected item schema used by the surrounding array field.",
        }
        if key in key_descriptions:
            return key_descriptions[key]
        if key.isdigit():
            return f"This block records the expected OpenAPI response entry for status code {key}."
        return f"This block gives the expected nested value for key `{key}` inside {words}."
    if block_type == "assert_rhs_block":
        if "snapshot(" in anchor:
            return f"This assertion snapshot is the expected structured value checked by {words}."
        return f"This assertion block is the expected response body checked by {words}."
    if block_type == "parametrize_arg_block":
        return f"This block lists the parameter cases used by {words}, including exact inputs and expected outputs."
    return f"This structured fragment is used by {words} and must be copied exactly."


def has_useful_fragility(source: str, target_lines: int) -> bool:
    features = block_features(source, target_lines)
    return (
        target_lines >= 18
        and int(features["punctuation_count"]) >= 100
        and int(features["quote_count"]) >= 12
    )


def candidate_span_from_key_value(
    lines: list[str],
    function_node: ast.AST,
    key: ast.AST | None,
    value: ast.AST,
) -> tuple[int, int, str, str] | None:
    if not is_string_constant(key):
        return None
    start = int(getattr(key, "lineno", 0))
    end = int(getattr(value, "end_lineno", 0))
    if not start or not end or end < start:
        return None
    text = source_lines(lines, start, end)
    anchor = clean_anchor(lines[start - 1])
    if not anchor or len(anchor) > 180:
        return None
    key_text = key_to_text(key)
    instruction_anchor = f"the block beginning with `{anchor}`"
    return start, end, text, f"{instruction_anchor} for key {key_text}"


def candidate_span_from_assert_rhs(lines: list[str], node: ast.Assert) -> tuple[int, int, str, str] | None:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return None
    rhs = test.comparators[0]
    if not isinstance(rhs, (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Call)):
        return None
    start = int(getattr(rhs, "lineno", 0))
    end = int(getattr(rhs, "end_lineno", 0))
    if not start or not end or end < start:
        return None
    text = source_lines(lines, start, end)
    anchor = clean_anchor(lines[int(getattr(node, "lineno", start)) - 1])
    if not anchor or len(anchor) > 180:
        return None
    return start, end, text, f"the assertion block beginning with `{anchor}`"


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def candidate_spans_from_decorator(lines: list[str], node: ast.AST) -> list[tuple[int, int, str, str]]:
    spans: list[tuple[int, int, str, str]] = []
    for dec in getattr(node, "decorator_list", []):
        if not isinstance(dec, ast.Call):
            continue
        name = call_name(dec.func)
        if "parametrize" not in name:
            continue
        for pos, arg in enumerate(dec.args):
            if not isinstance(arg, (ast.List, ast.Tuple)):
                continue
            start = int(getattr(arg, "lineno", 0))
            end = int(getattr(arg, "end_lineno", 0))
            if not start or not end or end < start:
                continue
            text = source_lines(lines, start, end)
            anchor = clean_anchor(lines[start - 1])
            if not anchor or len(anchor) > 180:
                continue
            ordinal = "first" if pos == 0 else "second" if pos == 1 else f"argument {pos + 1}"
            spans.append((start, end, text, f"the {ordinal} argument block passed to `@{name}`"))
    return spans


def collect_spans_for_function(
    lines: list[str],
    node: ast.AST,
    args: argparse.Namespace,
) -> list[tuple[int, int, str, str, str]]:
    spans: list[tuple[int, int, str, str, str]] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Dict):
            for key, value in zip(sub.keys, sub.values):
                if not isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Call)):
                    continue
                item = candidate_span_from_key_value(lines, node, key, value)
                if item:
                    start, end, text, instruction_anchor = item
                    spans.append((start, end, text, "key_value_block", instruction_anchor))
        elif isinstance(sub, ast.Assert):
            item = candidate_span_from_assert_rhs(lines, sub)
            if item:
                start, end, text, instruction_anchor = item
                spans.append((start, end, text, "assert_rhs_block", instruction_anchor))
    for start, end, text, instruction_anchor in candidate_spans_from_decorator(lines, node):
        spans.append((start, end, text, "parametrize_arg_block", instruction_anchor))

    filtered: list[tuple[int, int, str, str, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for start, end, text, block_type, instruction_anchor in spans:
        line_count = end - start + 1
        if not (args.min_target_lines <= line_count <= args.max_target_lines):
            continue
        function_start = int(getattr(node, "lineno", start))
        function_end = int(getattr(node, "end_lineno", end))
        function_lines = max(function_end - function_start + 1, 1)
        # Keep the task focused on an internal fragile span, not a near-whole
        # function that merely starts after a setup line.
        if line_count / function_lines > 0.85:
            continue
        if not has_useful_fragility(text, line_count):
            continue
        key = (start, end, block_type)
        if key in seen:
            continue
        seen.add(key)
        filtered.append((start, end, text, block_type, instruction_anchor))
    return filtered


def collect_blocks(args: argparse.Namespace) -> list[SpanBlock]:
    allow = {item.strip() for item in args.repo_allowlist.split(",") if item.strip()}
    blocks: list[SpanBlock] = []
    for path in sorted(args.data_root.rglob("*.py")):
        if not is_allowed_file(path):
            continue
        repo = repo_name(path, args.data_root)
        if allow and repo not in allow:
            continue
        text = read_text(path)
        if not text.strip():
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        functions = iter_function_nodes(tree)
        name_counts = Counter(qual[-1] for qual, _ in functions)
        rel = path.relative_to(args.data_root).as_posix()
        for qual_parts, node in functions:
            name = qual_parts[-1]
            if name_counts[name] != 1:
                continue
            function_text, function_start, function_end = function_source(lines, node)
            function_lines = function_end - function_start + 1
            if not (args.min_function_lines <= function_lines <= args.max_function_lines):
                continue
            context, ctx_start, ctx_end = build_context(
                lines,
                function_start,
                function_end,
                args.max_context_lines,
                args.min_context_lines,
                args.max_context_chars,
            )
            if function_text.strip() not in context:
                continue
            qualname = ".".join(qual_parts)
            for start, end, target, block_type, instruction_anchor in collect_spans_for_function(lines, node, args):
                target_lines = end - start + 1
                if target.strip() not in context:
                    continue
                if context.count(target.strip()) != 1:
                    continue
                anchor_text = clean_anchor(lines[start - 1])
                if context.count(anchor_text) != 1:
                    continue
                score = fragile_score(target, target_lines)
                blocks.append(
                    SpanBlock(
                        repo=repo,
                        relative_path=rel,
                        function_name=name,
                        qualname=qualname,
                        function_start_line=function_start,
                        function_end_line=function_end,
                        function_source=function_text,
                        context=context,
                        context_start_line=ctx_start,
                        context_end_line=ctx_end,
                        target_source=target,
                        target_start_line=start,
                        target_end_line=end,
                        target_lines=target_lines,
                        block_type=block_type,
                        anchor_text=anchor_text,
                        instruction_anchor=instruction_anchor,
                        explanation=make_explanation(block_type, name, anchor_text),
                        fragile_score=score,
                        features=block_features(target, target_lines),
                    )
                )
    return blocks


INSTRUCTION_TEMPLATES = (
    (
        "In the Python function `{target}`, briefly explain what the anchored block is used for, "
        "then quote exactly {anchor}, ending at the matching closing line for that block. "
        "Preserve every line, indentation level, quote, comma, bracket, and any trailing comma on the closing line."
    ),
    (
        "Look inside `{target}` and quote exactly {anchor}, stopping at the matching closing line. "
        "First give a brief explanation of the block's role. Do not include surrounding code outside that anchored block."
    ),
    (
        "For `{target}`, explain the purpose of the anchored structured block, then reproduce exactly {anchor}, "
        "ending at its matching close. "
        "Keep the copied block byte-for-byte the same apart from the surrounding fence, including a closing-line comma if present."
    ),
    (
        "From the excerpt, identify `{target}` and quote only {anchor}, through the matching closing line. "
        "Add a short explanation first, but keep the code block limited to the requested span."
    ),
)


def make_instruction(block: SpanBlock, variant: int = 0) -> str:
    target = block.qualname if "." in block.qualname else block.function_name
    return INSTRUCTION_TEMPLATES[variant % len(INSTRUCTION_TEMPLATES)].format(
        target=target,
        anchor=block.instruction_anchor,
    )


def make_context(block: SpanBlock) -> str:
    return (
        f"[FILE]\n{block.relative_path}\n"
        f"[LINES]\n{block.context_start_line}-{block.context_end_line}\n\n"
        "````python\n"
        f"{block.context.rstrip()}\n"
        "````"
    )


def make_gold(block: SpanBlock) -> str:
    return (
        "Explanation:\n"
        f"{block.explanation}\n\n"
        "Quoted block:\n"
        "````python\n"
        f"{block.target_source.rstrip()}\n"
        "````"
    )


def task_from_block(block: SpanBlock, split: str, variant: int = 0) -> Task:
    key = (
        f"{block.repo}:{block.relative_path}:{block.qualname}:"
        f"{block.block_type}:{block.target_start_line}:{block.target_end_line}"
    )
    variant_key = key if split == "val" else f"{key}:variant={variant}"
    digest = hashlib.sha1(variant_key.encode("utf-8")).hexdigest()[:10]
    suffix = "" if split == "val" else f".v{variant}"
    task_id = f"python_fragile_span.{block.repo}.{block.function_name}.{block.block_type}{suffix}.{digest}"
    return Task(
        task_id=task_id,
        case_id=variant_key,
        task_type="anchored_fragile_python_span",
        span_type=block.block_type,
        source_split=split,
        source_dataset="local-code-repos",
        instruction=make_instruction(block, variant),
        context=make_context(block),
        gold_answer=make_gold(block),
        gold_evidence=block.target_source,
        metadata={
            "repo": block.repo,
            "relative_path": block.relative_path,
            "function_name": block.function_name,
            "qualname": block.qualname,
            "function_start_line": block.function_start_line,
            "function_end_line": block.function_end_line,
            "context_start_line": block.context_start_line,
            "context_end_line": block.context_end_line,
            "context_lines": block.context_end_line - block.context_start_line + 1,
            "context_chars": len(block.context),
            "target_start_line": block.target_start_line,
            "target_end_line": block.target_end_line,
            "target_lines": block.target_lines,
            "target_chars": len(block.target_source),
            "block_type": block.block_type,
            "anchor_text": block.anchor_text,
            "instruction_anchor": block.instruction_anchor,
            "fragile_score": block.fragile_score,
            "fragile_features": block.features,
            "target_key": key,
            "instruction_variant": variant,
            "explanation_source": "agent_short_summary",
            "evaluation_target": "first_python_or_fenced_code_block",
        },
    )


def choose_val_blocks(blocks: list[SpanBlock], val_count: int) -> list[SpanBlock]:
    ordered = sorted(
        blocks,
        key=lambda b: (
            -b.fragile_score,
            -b.target_lines,
            stable_int(f"val:{b.repo}:{b.relative_path}:{b.qualname}:{b.target_start_line}"),
        ),
    )
    selected: list[SpanBlock] = []
    source_hashes: set[str] = set()
    by_path: Counter[str] = Counter()
    for block in ordered:
        if len(selected) >= val_count:
            break
        source_hash = hashlib.sha1(block.target_source.encode("utf-8")).hexdigest()
        if source_hash in source_hashes:
            continue
        if by_path[block.relative_path] >= 35:
            continue
        selected.append(block)
        source_hashes.add(source_hash)
        by_path[block.relative_path] += 1
    return selected


def make_train_tasks(
    blocks: list[SpanBlock],
    train_count: int,
    variants_per_block: int,
    seed: int,
) -> list[Task]:
    ordered = sorted(
        blocks,
        key=lambda b: (
            -b.fragile_score,
            -b.target_lines,
            stable_int(f"train:{b.repo}:{b.relative_path}:{b.qualname}:{b.target_start_line}:{seed}"),
        ),
    )
    tasks: list[Task] = []
    used_source_hashes: set[str] = set()
    for variant in range(max(1, variants_per_block)):
        for block in ordered:
            if len(tasks) >= train_count:
                return tasks
            source_hash = hashlib.sha1(block.target_source.encode("utf-8")).hexdigest()
            if variant == 0 and source_hash in used_source_hashes:
                continue
            tasks.append(task_from_block(block, "train", variant))
            used_source_hashes.add(source_hash)
    if len(tasks) < train_count:
        seen_cases = {task.case_id for task in tasks}
        for variant in range(1, max(2, variants_per_block)):
            for block in ordered:
                if len(tasks) >= train_count:
                    return tasks
                task = task_from_block(block, "train", variant)
                if task.case_id in seen_cases:
                    continue
                tasks.append(task)
                seen_cases.add(task.case_id)
    return tasks


def write_jsonl(path: Path, rows: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(round((len(values) - 1) * pct)))]


def stats_for(tasks: list[Task], candidate_count: int) -> dict[str, Any]:
    target_lines = [int(t.metadata["target_lines"]) for t in tasks]
    target_chars = [int(t.metadata["target_chars"]) for t in tasks]
    return {
        "candidate_blocks": candidate_count,
        "total_tasks": len(tasks),
        "train_tasks": sum(t.source_split == "train" for t in tasks),
        "val_tasks": sum(t.source_split == "val" for t in tasks),
        "train_unique_targets": len({t.metadata["target_key"] for t in tasks if t.source_split == "train"}),
        "val_unique_targets": len({t.metadata["target_key"] for t in tasks if t.source_split == "val"}),
        "repo_counts": dict(Counter(t.metadata["repo"] for t in tasks)),
        "span_type_counts": dict(Counter(t.span_type for t in tasks)),
        "target_line_stats": {
            "min": min(target_lines) if target_lines else 0,
            "median": percentile(target_lines, 0.5),
            "p90": percentile(target_lines, 0.9),
            "max": max(target_lines) if target_lines else 0,
        },
        "target_char_stats": {
            "min": min(target_chars) if target_chars else 0,
            "median": percentile(target_chars, 0.5),
            "p90": percentile(target_chars, 0.9),
            "max": max(target_chars) if target_chars else 0,
        },
        "explanation_source": "agent_short_summary",
        "evaluation_target": "first_python_or_fenced_code_block",
    }


def main() -> None:
    args = parse_args()
    blocks = collect_blocks(args)
    if not blocks:
        raise SystemExit("No candidate blocks found.")
    val_blocks = choose_val_blocks(blocks, min(args.val_count, len(blocks)))
    val_ids = {id(block) for block in val_blocks}
    val_source_hashes = {
        hashlib.sha1(block.target_source.encode("utf-8")).hexdigest() for block in val_blocks
    }
    train_blocks = [
        block
        for block in blocks
        if id(block) not in val_ids
        and hashlib.sha1(block.target_source.encode("utf-8")).hexdigest() not in val_source_hashes
    ]
    train_count = max(0, args.target_total - len(val_blocks))
    train_tasks = make_train_tasks(train_blocks, train_count, args.train_variants_per_block, args.seed)
    val_tasks = [task_from_block(block, "val", 0) for block in val_blocks]
    tasks = train_tasks + val_tasks

    write_jsonl(args.train_output, train_tasks)
    write_jsonl(args.val_output, val_tasks)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats_for(tasks, len(blocks)), ensure_ascii=False, indent=2) + "\n")
    print(f"candidates={len(blocks)} train={len(train_tasks)} val={len(val_tasks)} total={len(tasks)}")
    print(f"wrote train -> {args.train_output}")
    print(f"wrote val   -> {args.val_output}")
    print(f"wrote stats -> {args.stats_output}")


if __name__ == "__main__":
    main()
