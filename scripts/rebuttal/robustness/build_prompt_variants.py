#!/usr/bin/env python3
"""Build the prompt-robustness test sets: two instruction paraphrases and a tagged-output variant.

Only the instruction changes; context, gold answers and metadata are copied unchanged.

Example:
python -m scripts.rebuttal.robustness.build_prompt_variants \
  --data-dir data \
  --output-dir outputs/rebuttal/prompt_variants
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAINS = ("contract", "literature", "code")
VARIANTS = ("para_a", "para_b", "tagged")
OUTPUT_TAG = "COPYGEN_OUTPUT"
TAGGED_OUTPUT_REQUIREMENT = (
    "Format requirement: Put the entire requested answer unchanged between a "
    f"first line containing <{OUTPUT_TAG}> and a final line containing </{OUTPUT_TAG}>. "
    "Output no text outside those lines."
)

# Instruction templates of the Literature and Code test sets; the Literature request phrase
# is recovered by matching them.
LITERATURE_TEMPLATES = (
    (
        'From the paper "{paper}", extract the exact span from the subsection headed "{section}" '
        'that contains {request} and begins with "{anchor}". Output only the copied text.'
    ),
    (
        'In the paper "{paper}", return the exact text span from "{section}" '
        'that begins with "{anchor}" and provides {request}. Output only the copied text.'
    ),
    (
        'Extract the exact span from the subsection "{section}" in the paper "{paper}" '
        'that begins with "{anchor}" and describes {request}. Output only the copied text.'
    ),
)

# The test set uses the first two Code templates; the paraphrases use the other two.
CODE_TEMPLATES = (
    (
        "In the Python function `{target}`, briefly explain what the anchored block is used for, "
        "then quote exactly {anchor}, ending at the matching closing line for that block. "
        "Preserve every line, indentation level, quote, comma, bracket, and any trailing comma "
        "on the closing line."
    ),
    (
        "Look inside `{target}` and quote exactly {anchor}, stopping at the matching closing line. "
        "First give a brief explanation of the block's role. Do not include surrounding code "
        "outside that anchored block."
    ),
    (
        "For `{target}`, explain the purpose of the anchored structured block, then reproduce "
        "exactly {anchor}, ending at its matching close. Keep the copied block byte-for-byte the "
        "same apart from the surrounding fence, including a closing-line comma if present."
    ),
    (
        "From the excerpt, identify `{target}` and quote only {anchor}, through the matching "
        "closing line. Add a short explanation first, but keep the code block limited to the "
        "requested span."
    ),
)


def _quote(value: str) -> str:
    return value.replace('"', '\\"')


def _literature_request(row: dict) -> str:
    metadata = row["metadata"]
    fields = {
        "paper": _quote(metadata["paper_title"]),
        "section": _quote(metadata["section_title"]),
        "anchor": _quote(metadata["anchor_text"]),
    }
    marker = "\0"
    instruction = row["instruction"]
    matches = []
    for template in LITERATURE_TEMPLATES:
        prefix, suffix = template.format(request=marker, **fields).split(marker)
        if not instruction.startswith(prefix) or not instruction.endswith(suffix):
            continue
        request = instruction[len(prefix) : len(instruction) - len(suffix)]
        if request and template.format(request=request, **fields) == instruction:
            matches.append(request)
    if len(matches) != 1:
        raise ValueError(f"{row['task_id']}: instruction matches {len(matches)} literature templates")
    return matches[0]


def literature_paraphrase(row: dict, variant: str) -> str:
    metadata = row["metadata"]
    paper = _quote(metadata["paper_title"])
    section = _quote(metadata["section_title"])
    anchor = _quote(metadata["anchor_text"])
    request = _literature_request(row)
    if variant == "para_a":
        return (
            "VERBATIM SPAN TASK\n"
            f'Paper: "{paper}"\n'
            f'Subsection: "{section}"\n'
            f"Requested content: {request}\n"
            f'Start text: "{anchor}"\n'
            "Copy the complete requested span beginning with the Start text. "
            "Preserve the source exactly and return no other text."
        )
    return (
        f'Find the source span corresponding to "{_quote(request)}" in subsection '
        f'"{section}" of "{paper}". It begins exactly with "{anchor}". '
        "Return that entire span verbatim, with no preface or markup."
    )


def contract_paraphrase(row: dict, variant: str) -> str:
    metadata = row["metadata"]
    mappings = [
        (_quote(key), _quote(title))
        for key, title in zip(metadata["selected_clause_keys"], metadata["selected_clause_titles"])
    ]
    preservation = "wording, line breaks, and indentation" if metadata["preserve_clause_layout"] else "wording"
    if variant == "para_a":
        numbered = "\n".join(f'{index}. "{key}" <- "{title}"' for index, (key, title) in enumerate(mappings, start=1))
        return (
            "VERBATIM CLAUSE TASK\n"
            "Create one JSON object from the article.\n"
            "Ordered key-to-heading mappings:\n"
            f"{numbered}\n"
            "For every mapping, copy the complete clause including its heading as the value. "
            f"Keep this key order and preserve the original {preservation}. "
            "Return only the JSON object."
        )
    fields = "; ".join(f'"{key}" with the complete clause headed "{title}"' for key, title in mappings)
    return (
        "Using only the article, fill these JSON fields in the listed order: "
        f"{fields}. Copy every heading and clause body exactly, preserving the original "
        f"{preservation}. Return one JSON object and no other text."
    )


def code_paraphrase(row: dict, variant: str) -> str:
    metadata = row["metadata"]
    qualname = metadata["qualname"]
    target = qualname if "." in qualname else metadata["function_name"]
    template = CODE_TEMPLATES[2 if variant == "para_a" else 3]
    return template.format(target=target, anchor=metadata["instruction_anchor"])


PARAPHRASES = {
    "literature": literature_paraphrase,
    "contract": contract_paraphrase,
    "code": code_paraphrase,
}


def variant_instruction(row: dict, domain: str, variant: str) -> str:
    if variant == "tagged":
        return f"{row['instruction']}\n\n{TAGGED_OUTPUT_REQUIREMENT}"
    return PARAPHRASES[domain](row, variant)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paraphrased and tagged-output prompt variants.")
    parser.add_argument("--data-dir", default="data", help="Directory with <domain>/test.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        with (Path(args.data_dir) / domain / "test.jsonl").open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for variant in VARIANTS:
            out_path = output_dir / f"{domain}.{variant}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    variant_row = dict(row)
                    variant_row["instruction"] = variant_instruction(row, domain, variant)
                    f.write(json.dumps(variant_row, ensure_ascii=False) + "\n")
            print(f"saved {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
