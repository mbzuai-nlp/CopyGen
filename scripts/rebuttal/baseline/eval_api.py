#!/usr/bin/env python3
"""
Query a chat model through an OpenAI-compatible API on the copy benchmark (temperature 0).

Each response is appended to <output-dir>/<domain>/predictions.jsonl as soon as it arrives;
task_ids already in that file are skipped, so an interrupted run is resumed by rerunning the
same command. `prediction` is the response with at most one outer Markdown code fence removed.

Example:
  python scripts/rebuttal/baseline/eval_api.py \
    --model deepseek-v4-pro \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --output-dir outputs/rebuttal/closed_api/deepseek_v4_pro
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.prompt_utils import build_prompt_messages

DOMAINS = ("contract", "literature", "code", "random_sequence")
OUTER_FENCE_RE = re.compile(
    r"\A```(?P<info>[^`\r\n]*)\r?\n(?P<body>.*?)\r?\n```[ \t\r\n]*\Z",
    flags=re.DOTALL,
)


def strip_outer_fence(text: str) -> str:
    """Remove one ``` fence that wraps the whole response; partial or inner fences are kept."""
    match = OUTER_FENCE_RE.fullmatch(text)
    return match.group("body") if match else text


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def query(client: AsyncOpenAI, semaphore: asyncio.Semaphore, row: dict, args: argparse.Namespace) -> dict:
    extra = {"extra_body": {"thinking": {"type": "disabled"}}} if args.disable_thinking else {}
    async with semaphore:
        start_time = time.monotonic()
        response = await client.chat.completions.create(
            model=args.model,
            messages=build_prompt_messages(row["context"], row["instruction"]),
            temperature=0.0,
            max_tokens=args.max_tokens,
            **extra,
        )
        latency = time.monotonic() - start_time
    choice = response.choices[0]
    text = choice.message.content or ""
    return {
        "task_id": row["task_id"],
        "prediction": strip_outer_fence(text),
        "response": text,
        "response_model": response.model,
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump(exclude_none=True) if response.usage else {},
        "latency_seconds": latency,
    }


async def run(args: argparse.Namespace) -> int:
    client = AsyncOpenAI(
        api_key=os.environ[args.api_key_env],
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = 0
    for domain in args.domains:
        rows = load_rows(Path(args.data_root) / domain / "test.jsonl")
        output_path = Path(args.output_dir) / domain / "predictions.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        done = {row["task_id"] for row in load_rows(output_path)} if output_path.exists() else set()
        pending = [row for row in rows if row["task_id"] not in done]
        print(f"{domain}: {len(done)} done, {len(pending)} to query", flush=True)

        with output_path.open("a", encoding="utf-8") as f:
            tasks = [asyncio.ensure_future(query(client, semaphore, row, args)) for row in pending]
            for finished, task in enumerate(asyncio.as_completed(tasks), start=1):
                try:
                    record = await task
                except Exception as error:
                    failures += 1
                    print(f"{domain}: request failed ({type(error).__name__}): {error}", flush=True)
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if finished % 20 == 0 or finished == len(tasks):
                    print(f"{domain}: {finished}/{len(tasks)}", flush=True)
    await client.close()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an OpenAI-compatible chat model on the copy benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the API key.")
    parser.add_argument("--data-root", default="data", help="Directory with <domain>/test.jsonl.")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send thinking={'type': 'disabled'}; use --no-disable-thinking for models without a thinking mode.",
    )
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=8)
    args = parser.parse_args()

    failures = asyncio.run(run(args))
    if failures:
        raise SystemExit(f"{failures} requests failed; rerun the same command to fill them in.")


if __name__ == "__main__":
    main()
