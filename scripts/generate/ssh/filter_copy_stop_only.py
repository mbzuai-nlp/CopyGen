#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def action_modes(row: dict) -> list[str]:
    return [action.get("mode", "") for action in row.get("actions", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only oracle rows with COPY+STOP actions.")
    parser.add_argument("--oracle-input", required=True, help="Input oracle JSONL")
    parser.add_argument("--oracle-output", required=True, help="Filtered oracle JSONL")
    parser.add_argument("--task-input", default="", help="Optional original task JSONL to filter by task_id")
    parser.add_argument("--task-output", default="", help="Optional filtered task JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum kept rows")
    args = parser.parse_args()

    oracle_rows = load_jsonl(Path(args.oracle_input))
    kept_oracle = [row for row in oracle_rows if action_modes(row) == ["COPY", "STOP"]]
    if args.limit > 0:
        kept_oracle = kept_oracle[: args.limit]

    oracle_output = Path(args.oracle_output)
    oracle_output.parent.mkdir(parents=True, exist_ok=True)
    with oracle_output.open("w", encoding="utf-8") as f:
        for row in kept_oracle:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"kept_oracle={len(kept_oracle)}")
    print(f"oracle_output={oracle_output}")

    if args.task_input and args.task_output:
        keep_ids = {row.get("task_id") for row in kept_oracle}
        task_rows = load_jsonl(Path(args.task_input))
        kept_tasks = [row for row in task_rows if row.get("task_id") in keep_ids]
        task_output = Path(args.task_output)
        task_output.parent.mkdir(parents=True, exist_ok=True)
        with task_output.open("w", encoding="utf-8") as f:
            for row in kept_tasks:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"kept_tasks={len(kept_tasks)}")
        print(f"task_output={task_output}")


if __name__ == "__main__":
    main()
