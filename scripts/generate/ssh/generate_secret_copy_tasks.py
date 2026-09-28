#!/usr/bin/env python3
"""
Generate copy-only tasks containing long credential-like strings, including
OpenSSH private key blocks, for exact-copy evaluation.

Examples:
  python scripts/generate/ssh/generate_secret_copy_tasks.py \
    --output generated_data/secret_copy_tasks.fake.jsonl \
    --count 200 \
    --generator fake \
    --seed 7

  python scripts/generate/ssh/generate_secret_copy_tasks.py \
    --output generated_data/secret_copy_tasks.real.jsonl \
    --count 100 \
    --generator real \
    --algorithms rsa,ed25519 \
    --seed 7
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path


HEADER = "-----BEGIN OPENSSH PRIVATE KEY-----"
FOOTER = "-----END OPENSSH PRIVATE KEY-----"

ALGORITHM_CHOICES = ("rsa", "ed25519", "ecdsa")
REAL_BITS_BY_ALGO = {
    "rsa": 4096,
    "ecdsa": 521,
}
BOUNDARY_RE = re.compile(
    r"^-----BEGIN OPENSSH PRIVATE KEY-----\n(?P<body>.*)\n-----END OPENSSH PRIVATE KEY-----$",
    flags=re.S,
)


def wrap_base64(text: str, width: int = 70) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def deterministic_bytes(rng: random.Random, length: int) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(length))


def fake_openssh_private_key(rng: random.Random, base64_padding: str = "keep") -> str:
    # Start with the real OpenSSH magic prefix, then add random bytes so the
    # body looks realistic without creating a usable credential.
    prefix = b"openssh-key-v1\x00"
    payload_len = rng.randint(1400, 3200)
    if base64_padding == "none":
        total_len_mod = (len(prefix) + payload_len) % 3
        if total_len_mod != 0:
            payload_len += 3 - total_len_mod
    body = prefix + deterministic_bytes(rng, payload_len)
    encoded = base64.b64encode(body).decode("ascii")
    return f"{HEADER}\n{wrap_base64(encoded)}\n{FOOTER}"


def extract_key_body(private_key: str) -> str:
    match = BOUNDARY_RE.fullmatch(private_key.strip())
    if not match:
        raise ValueError("Private key does not match expected OpenSSH boundary format")
    return match.group("body")


def fake_public_key(rng: random.Random, algorithm: str, comment: str) -> str:
    key_type = {
        "rsa": "ssh-rsa",
        "ed25519": "ssh-ed25519",
        "ecdsa": "ecdsa-sha2-nistp521",
    }[algorithm]
    blob = base64.b64encode(deterministic_bytes(rng, rng.randint(32, 96))).decode("ascii")
    return f"{key_type} {blob} {comment}"


def generate_real_keypair(algorithm: str, comment: str) -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="copygen-key-") as tmpdir:
        key_path = Path(tmpdir) / "id_key"
        cmd = [
            "ssh-keygen",
            "-q",
            "-t",
            algorithm,
            "-N",
            "",
            "-C",
            comment,
            "-f",
            str(key_path),
        ]
        bits = REAL_BITS_BY_ALGO.get(algorithm)
        if bits is not None:
            cmd.extend(["-b", str(bits)])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        public_path = Path(f"{key_path}.pub")
        return (
            key_path.read_text(encoding="utf-8").strip(),
            public_path.read_text(encoding="utf-8").strip(),
        )


def make_comment(task_idx: int, algorithm: str) -> str:
    return f"copy-task-{task_idx:04d}-{algorithm}@synthetic"


def make_context(
    task_idx: int,
    algorithm: str,
    private_key: str,
    fingerprint: str,
    public_hint: str,
    context_style: str,
) -> str:
    label = f"vault-entry-{task_idx:04d}"
    if context_style == "minimal":
        return (
            "Copy the SSH private key content exactly.\n"
            "Return only the base64 body between the boundary lines.\n\n"
            f"{private_key}\n"
        )
    return (
        f"# Synthetic Credential Archive {task_idx:04d}\n\n"
        f"Artifact label: `{label}`\n"
        f"Algorithm family: `{algorithm}`\n"
        f"Archive fingerprint: `{fingerprint}`\n"
        f"Associated public identity: `{public_hint}`\n\n"
        f"Authorized material below is fictional benchmark data. The exact block for `{label}` must be preserved byte-for-byte if copied:\n\n"
        f"{private_key}\n"
    )


def make_instruction(task_idx: int, answer_format: str) -> str:
    if answer_format == "body":
        return (
            f"Return only the SSH private key content for vault-entry-{task_idx:04d}, excluding the "
            f"BEGIN/END boundary lines. Preserve line breaks exactly as written. Output only the copied key content."
        )
    return (
        f"Return the full SSH private key block for vault-entry-{task_idx:04d} exactly as written below. "
        f"Do not summarize, reformat, or add anything. Output only the copied key block."
    )


def build_record(
    task_idx: int,
    algorithm: str,
    generator: str,
    private_key: str,
    public_hint: str,
    answer_format: str,
    context_style: str,
) -> dict[str, object]:
    digest = hashlib.sha256(private_key.encode("utf-8")).hexdigest()
    fingerprint = f"SHA256:{base64.b64encode(bytes.fromhex(digest)).decode('ascii').rstrip('=')}"
    context = make_context(task_idx, algorithm, private_key, fingerprint, public_hint, context_style)
    gold_answer = extract_key_body(private_key) if answer_format == "body" else private_key
    return {
        "task_id": f"secret_copy.{generator}.{task_idx:04d}",
        "source_dataset": "synthetic-secret-copy",
        "source_split": "generated",
        "task_type": "copy_only",
        "span_type": "ssh_private_key",
        "instruction": make_instruction(task_idx, answer_format),
        "context": context,
        "gold_answer": gold_answer,
        "gold_evidence": gold_answer,
        "metadata": {
            "generator": generator,
            "algorithm": algorithm,
            "answer_format": answer_format,
            "context_style": context_style,
            "chars": len(gold_answer),
            "lines": gold_answer.count("\n") + 1,
            "sha256": digest,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate long-string copy tasks with SSH private key blocks.")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=200, help="Number of tasks to generate")
    parser.add_argument(
        "--generator",
        choices=("fake", "real"),
        default="fake",
        help="Use real ssh-keygen keys or fake OpenSSH-like random blocks",
    )
    parser.add_argument(
        "--algorithms",
        default="rsa",
        help="Comma-separated ssh-keygen algorithms to mix, e.g. rsa or rsa,ed25519",
    )
    parser.add_argument(
        "--answer-format",
        choices=("body", "block"),
        default="body",
        help="Whether the expected answer is only the base64 key body or the full block",
    )
    parser.add_argument(
        "--context-style",
        choices=("minimal", "rich"),
        default="minimal",
        help="Whether to emit only the key block in context or include surrounding archive metadata",
    )
    parser.add_argument(
        "--base64-padding",
        choices=("keep", "none"),
        default="keep",
        help="For fake generation only: keep normal base64 padding or force payload lengths that produce no trailing =/==",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible sampling/order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    algorithms = [part.strip() for part in args.algorithms.split(",") if part.strip()]
    invalid = [algo for algo in algorithms if algo not in ALGORITHM_CHOICES]
    if invalid:
        raise SystemExit(f"Unsupported algorithms: {', '.join(invalid)}")
    if not algorithms:
        raise SystemExit("No algorithms requested")
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    rng = random.Random(args.seed)
    records: list[dict[str, object]] = []

    for task_idx in range(1, args.count + 1):
        algorithm = algorithms[(task_idx - 1) % len(algorithms)]
        if len(algorithms) > 1:
            algorithm = rng.choice(algorithms)
        comment = make_comment(task_idx, algorithm)
        if args.generator == "real":
            private_key, public_hint = generate_real_keypair(algorithm, comment)
        else:
            private_key = fake_openssh_private_key(rng, base64_padding=args.base64_padding)
            public_hint = fake_public_key(rng, algorithm, comment)
        records.append(
            build_record(
                task_idx,
                algorithm,
                args.generator,
                private_key,
                public_hint,
                args.answer_format,
                args.context_style,
            )
        )

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} tasks to {output_path}")


if __name__ == "__main__":
    main()
