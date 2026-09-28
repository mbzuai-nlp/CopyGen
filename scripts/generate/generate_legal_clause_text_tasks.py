#!/usr/bin/env python3
"""
Transform curated legal clause bundle tasks into single-clause pure-text copy tasks.

Example:
  conda run -n mad python scripts/generate/generate_legal_clause_text_tasks.py \
    --train-input outputs/generated/legal_clause/legal_clause_bundle_tasks.train.curated.jsonl \
    --val-input outputs/generated/legal_clause/legal_clause_bundle_tasks.val.curated.jsonl \
    --train-output generated_data/legal_clause_text_copy_tasks.train.jsonl \
    --val-output generated_data/legal_clause_text_copy_tasks.val.jsonl \
    --stats-output generated_data/legal_clause_text_copy_tasks.stats.json \
    --min-clause-chars 700 \
    --instruction-mode anchored
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


KEY_TO_INSTRUCTION = {
    "governing_law_clause": (
        "Return the full clause that states what law governs the agreement and, if present, any fallback rule "
        "for choosing the governing law. Copy only that clause, including its heading."
    ),
    "consent_to_jurisdiction_clause": (
        "Return the full clause where the parties submit to specified courts, consent to jurisdiction or venue, "
        "and waive objections based on forum or inconvenience. Copy only that clause, including its heading."
    ),
    "waiver_of_jury_trial_clause": (
        "Return the full clause where the parties waive any right to a jury trial. Copy only that clause, "
        "including its heading."
    ),
    "service_of_process_clause": (
        "Return the full clause describing how legal process may be served and what methods of service are allowed. "
        "Copy only that clause, including its heading."
    ),
    "notices_clause": (
        "Return the full clause specifying how notices and other communications must be sent, including delivery "
        "methods and timing rules. Copy only that clause, including its heading."
    ),
    "severability_clause": (
        "Return the full clause explaining what happens if part of the agreement is invalid, illegal, or "
        "unenforceable. Copy only that clause, including its heading."
    ),
    "entire_agreement_clause": (
        "Return the full clause stating that the written agreement and related documents are the complete agreement "
        "between the parties and supersede prior understandings. Copy only that clause, including its heading."
    ),
    "amendments_and_waivers_clause": (
        "Return the full clause governing how the agreement may be amended or waived and when waiver language is "
        "effective. Copy only that clause, including its heading."
    ),
    "successors_and_assigns_clause": (
        "Return the full clause describing who is bound by the agreement after assignment, succession, merger, or "
        "transfer. Copy only that clause, including its heading."
    ),
    "counterparts_clause": (
        "Return the full clause stating that the agreement may be signed in counterparts and that electronic "
        "delivery or signature is effective. Copy only that clause, including its heading."
    ),
    "waiver_of_venue_clause": (
        "Return the full clause where the parties waive objections to venue or forum non conveniens. Copy only that "
        "clause, including its heading."
    ),
    "confidentiality_clause": (
        "Return the full clause that restricts disclosure of confidential information or publicity-related "
        "information. Copy only that clause, including its heading."
    ),
    "expenses_clause": (
        "Return the full clause allocating fees, costs, expenses, or reimbursement obligations. Copy only that "
        "clause, including its heading."
    ),
    "nonreliance_clause": (
        "Return the full clause stating that the parties are not relying on extra-contractual statements, "
        "representations, or promises. Copy only that clause, including its heading."
    ),
}

EXCLUDED_CLAUSE_KEYS = {
    "nonreliance_clause",
}

TRAILING_PUNCTUATION = set(".;:!?)]}\"'”’")
LOW_QUALITY_PATTERNS = [
    re.compile(r"remainder of page", re.IGNORECASE),
    re.compile(r"left blank", re.IGNORECASE),
    re.compile(r"signatures follow", re.IGNORECASE),
    re.compile(r"\bsignature\s*:", re.IGNORECASE),
    re.compile(r"\bseller\s*:", re.IGNORECASE),
    re.compile(r"\bbuyer\s*:", re.IGNORECASE),
    re.compile(r"\boptionee\b", re.IGNORECASE),
    re.compile(r"executed this agreement", re.IGNORECASE),
    re.compile(r"acknowledges having received a copy", re.IGNORECASE),
    re.compile(r"important notice:", re.IGNORECASE),
    re.compile(r"membership interest purchase agreement", re.IGNORECASE),
    re.compile(r"micron confidential", re.IGNORECASE),
    re.compile(r"confidential green\b", re.IGNORECASE),
    re.compile(r"registered or certified\.\s*$", re.IGNORECASE),
    re.compile(r"\[\s*\*+\s*\]"),
    re.compile(r"has been omitted", re.IGNORECASE),
]
INCOMPLETE_ENDINGS = {
    "of",
    "on",
    "to",
    "for",
    "and",
    "or",
    "by",
    "with",
    "under",
    "into",
    "from",
    "upon",
}
OCR_SPLIT_PATTERNS = [
    re.compile(r"\b[A-Za-z]{3,}\s+(?![aAI]\b)[A-Za-z]\b"),
    re.compile(r"\b(?![aAI]\b)[A-Za-z]\s+[A-Za-z]{3,}\b"),
]
ARTICLE_TITLE_DROP_PATTERNS = [
    re.compile(r"^\([a-z]\)\s+if\s+to\b", re.IGNORECASE),
    re.compile(r"\b(attn|attention|facsimile|telephone|telecopy|email|enterprise\s+ln|street|avenue|boulevard|road|drive|suite)\b", re.IGNORECASE),
]
ENUM_PATTERN = re.compile(r"\(([a-z]|[ivx]+)\)")
XREF_PATTERN = re.compile(r"\b(?:Section|Sections|Article|Articles)\s+[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*", re.IGNORECASE)
ALLCAPS_WORD_PATTERN = re.compile(r"\b[A-Z][A-Z0-9/-]{2,}\b")
HARD_CLAUSE_KEYS = {
    "governing_law_clause",
    "consent_to_jurisdiction_clause",
    "waiver_of_jury_trial_clause",
    "waiver_of_venue_clause",
    "severability_clause",
    "entire_agreement_clause",
    "counterparts_clause",
    "amendments_and_waivers_clause",
    "service_of_process_clause",
    "notices_clause",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("outputs/generated/legal_clause/legal_clause_bundle_tasks.train.curated.jsonl"),
    )
    parser.add_argument(
        "--val-input",
        type=Path,
        default=Path("outputs/generated/legal_clause/legal_clause_bundle_tasks.val.curated.jsonl"),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("generated_data/legal_clause_text_copy_tasks.train.jsonl"),
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("generated_data/legal_clause_text_copy_tasks.val.jsonl"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("generated_data/legal_clause_text_copy_tasks.stats.json"),
    )
    parser.add_argument("--min-clause-chars", type=int, default=500)
    parser.add_argument("--max-train-tasks", type=int, default=0)
    parser.add_argument("--max-val-tasks", type=int, default=0)
    parser.add_argument(
        "--instruction-mode",
        choices=["realistic", "anchored"],
        default="realistic",
        help="Use natural semantic requests ('realistic') or uniquely anchored clause requests ('anchored').",
    )
    parser.add_argument(
        "--difficulty",
        choices=["all", "hard"],
        default="all",
        help="Keep all valid clauses or only a harder subset with long, structurally complex clauses.",
    )
    parser.add_argument(
        "--min-hard-score",
        type=int,
        default=4,
        help="Minimum hard-score required when --difficulty hard is used.",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=["source", "instruction_gold", "exact"],
        default="source",
        help=(
            "How aggressively to deduplicate tasks after basic filtering: "
            "'source' keeps only source-level deduplication, "
            "'instruction_gold' deduplicates tasks with identical instruction and target clause, "
            "and 'exact' deduplicates only fully identical instruction/target/context triples."
        ),
    )
    parser.add_argument(
        "--train-dedupe-mode",
        choices=["source", "instruction_gold", "exact"],
        default=None,
        help="Optional override for the training split. If omitted, uses --dedupe-mode.",
    )
    parser.add_argument(
        "--val-dedupe-mode",
        choices=["source", "instruction_gold", "exact"],
        default=None,
        help="Optional override for the validation split. If omitted, uses --dedupe-mode.",
    )
    parser.add_argument(
        "--allow-train-low-quality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, keep train clauses that would normally be filtered by the low-quality heuristic.",
    )
    parser.add_argument(
        "--allow-val-low-quality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, keep val clauses that would normally be filtered by the low-quality heuristic.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


CLAUSE_NUMBER_RE = re.compile(r"^\s*([0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)\.\s+")


def extract_clause_number(clause_text: str) -> str:
    first_line = clause_text.splitlines()[0].strip() if clause_text.strip() else ""
    match = CLAUSE_NUMBER_RE.match(first_line)
    if not match:
        return ""
    return match.group(1)


def _clean_clause_heading(clause_title: str) -> str:
    return clause_title.strip()


def _normalize_ocr_split_caps(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    previous = None
    while normalized != previous:
        previous = normalized
        normalized = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", normalized)
        normalized = re.sub(r"\b([A-Z]{2,})\s+([A-Z])\b", r"\1\2", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def sanitize_article_title(article_title: str) -> str:
    title = _normalize_ocr_split_caps(article_title)
    if not title:
        return ""
    for pattern in ARTICLE_TITLE_DROP_PATTERNS:
        if pattern.search(title):
            return ""
    if title.startswith("("):
        return ""
    return title.strip(" ;,.-")


def make_instruction(
    clause_key: str,
    clause_title: str,
    clause_text: str,
    article_label: str,
    article_title: str,
    instruction_mode: str,
) -> str:
    if instruction_mode == "anchored":
        clause_number = extract_clause_number(clause_text)
        clause_heading = _clean_clause_heading(clause_title)
        clean_article_label = article_label.strip()
        clean_article_title = sanitize_article_title(article_title)
        if clause_number:
            article_fragment = ""
            if clean_article_label:
                article_fragment = f" from Article {clean_article_label}"
                if clean_article_title:
                    article_fragment += f', titled "{clean_article_title}"'
            elif clean_article_title:
                article_fragment = f' from the article titled "{clean_article_title}"'
            return (
                f"From the agreement excerpt below, reproduce the full text of Section {clause_number}, "
                f"headed \"{clause_heading}\"{article_fragment}. Include the section heading exactly as written. "
                "Output only that clause."
            )
        return (
            f"From the agreement excerpt below, reproduce the full text of the clause headed "
            f"\"{clause_heading}\". Include the section heading exactly as written. Output only that clause."
        )

    template = KEY_TO_INSTRUCTION.get(clause_key)
    if template:
        return template + " Output only the copied clause."
    return (
        f"Return the full clause whose topic is <<{clause_title}>>. "
        "Copy it exactly as written, including its heading. Output only the copied clause."
    )


def is_low_quality_clause(clause_text: str) -> bool:
    stripped = clause_text.strip()
    lowered = stripped.lower()
    for pattern in LOW_QUALITY_PATTERNS:
        if pattern.search(stripped):
            return True
    ocr_split_hits = sum(len(pattern.findall(stripped)) for pattern in OCR_SPLIT_PATTERNS)
    if ocr_split_hits >= 4:
        return True
    if stripped and stripped[-1] not in TRAILING_PUNCTUATION:
        last_word = re.findall(r"[A-Za-z]+", lowered)
        if not last_word:
            return True
        if last_word[-1] in INCOMPLETE_ENDINGS:
            return True
        return True
    return False


def compute_hard_features(
    clause_text: str,
    clause_key: str,
    article_label: str,
    article_title: str,
) -> tuple[int, dict]:
    text = clause_text.strip()
    answer_chars = len(text)
    enum_count = len(ENUM_PATTERN.findall(text))
    xref_count = len(XREF_PATTERN.findall(text))
    semicolon_count = text.count(";")
    allcaps_count = len(ALLCAPS_WORD_PATTERN.findall(text))
    has_article_heading = bool(article_label or article_title)
    hard_key = clause_key in HARD_CLAUSE_KEYS

    score = 0
    if answer_chars >= 1200:
        score += 2
    elif answer_chars >= 800:
        score += 1
    if enum_count >= 2:
        score += 1
    if xref_count >= 2:
        score += 1
    if semicolon_count >= 3:
        score += 1
    if allcaps_count >= 10:
        score += 1
    if has_article_heading:
        score += 1
    if hard_key:
        score += 1

    return score, {
        "hard_score": score,
        "answer_chars": answer_chars,
        "enum_count": enum_count,
        "xref_count": xref_count,
        "semicolon_count": semicolon_count,
        "allcaps_count": allcaps_count,
        "has_article_heading": has_article_heading,
        "hard_key_family": hard_key,
    }


def build_split(
    input_path: Path,
    output_path: Path,
    split_name: str,
    min_clause_chars: int,
    max_tasks: int,
    instruction_mode: str,
    difficulty: str,
    min_hard_score: int,
    dedupe_mode: str,
    allow_low_quality: bool,
) -> dict:
    rows_out = []
    seen_source = set()
    seen_task = set()
    too_short = 0
    source_duplicates = 0
    task_duplicates = 0
    not_verbatim = 0
    low_quality = 0
    not_hard = 0
    bundles = 0
    clause_lengths = []
    clause_key_counts: Counter[str] = Counter()
    hard_scores: list[int] = []

    for bundle in iter_jsonl(input_path):
        bundles += 1
        answers = json.loads(bundle["gold_answer"])
        meta = bundle.get("metadata", {})
        clause_titles = dict(
            zip(
                meta.get("selected_clause_keys", []),
                meta.get("selected_clause_titles", []),
            )
        )
        text_path = meta.get("text_path", "")
        article_label = meta.get("article_label", "")
        article_title = meta.get("article_title", "")
        clean_article_title = sanitize_article_title(article_title)

        for clause_key, clause_text in answers.items():
            if clause_key in EXCLUDED_CLAUSE_KEYS:
                continue
            if len(clause_text) < min_clause_chars:
                too_short += 1
                continue

            clause_title = clause_titles.get(clause_key, clause_key)
            fingerprint = hashlib.sha1(clause_text.encode("utf-8")).hexdigest()[:16]
            source_dedupe_key = (text_path, article_label, clause_key, fingerprint)
            if source_dedupe_key in seen_source:
                source_duplicates += 1
                continue
            if bundle["context"].count(clause_text) != 1:
                not_verbatim += 1
                continue
            if not allow_low_quality and is_low_quality_clause(clause_text):
                low_quality += 1
                continue
            hard_score, hard_features = compute_hard_features(
                clause_text,
                clause_key,
                article_label,
                clean_article_title,
            )
            if difficulty == "hard" and hard_score < min_hard_score:
                not_hard += 1
                continue

            instruction = make_instruction(
                clause_key,
                clause_title,
                clause_text,
                article_label,
                clean_article_title,
                instruction_mode,
            )
            if dedupe_mode == "instruction_gold":
                task_dedupe_key = (instruction, clause_text)
            elif dedupe_mode == "exact":
                task_dedupe_key = (instruction, clause_text, bundle["context"])
            else:
                task_dedupe_key = source_dedupe_key
            if task_dedupe_key in seen_task:
                task_duplicates += 1
                continue

            seen_source.add(source_dedupe_key)
            seen_task.add(task_dedupe_key)

            task_id = f"legal_clause_text.{split_name}.{len(rows_out) + 1:06d}"
            rows_out.append(
                {
                    "task_id": task_id,
                    "source_dataset": bundle.get("source_dataset", "edgar_contracts"),
                    "source_split": split_name,
                    "task_type": "legal_clause_copy",
                    "span_type": "legal_clause",
                    "instruction": instruction,
                    "context": bundle["context"],
                    "gold_answer": clause_text,
                    "gold_evidence": clause_text,
                    "metadata": {
                        "source_bundle_task_id": bundle["task_id"],
                        "source_case_id": bundle.get("case_id"),
                        "clause_key": clause_key,
                        "clause_title": clause_title,
                        "clause_number": extract_clause_number(clause_text),
                        "instruction_mode": instruction_mode,
                        "text_path": text_path,
                        "article_label": article_label,
                        "article_title": clean_article_title,
                        "article_title_raw": article_title,
                        "window_section_numbers": meta.get("window_section_numbers", []),
                        "context_chars": len(bundle["context"]),
                        "answer_chars": len(clause_text),
                        **hard_features,
                    },
                }
            )
            clause_lengths.append(len(clause_text))
            clause_key_counts[clause_key] += 1
            hard_scores.append(hard_score)
            if max_tasks and len(rows_out) >= max_tasks:
                break
        if max_tasks and len(rows_out) >= max_tasks:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if clause_lengths:
        clause_lengths.sort()
        summary = {
            "min": clause_lengths[0],
            "median": clause_lengths[len(clause_lengths) // 2],
            "max": clause_lengths[-1],
            "mean": sum(clause_lengths) / len(clause_lengths),
        }
    else:
        summary = {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    if hard_scores:
        sorted_scores = sorted(hard_scores)
        hard_summary = {
            "min": sorted_scores[0],
            "median": sorted_scores[len(sorted_scores) // 2],
            "max": sorted_scores[-1],
            "mean": sum(sorted_scores) / len(sorted_scores),
        }
    else:
        hard_summary = {"min": 0, "median": 0, "max": 0, "mean": 0.0}

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "instruction_mode": instruction_mode,
        "difficulty": difficulty,
        "dedupe_mode": dedupe_mode,
        "allow_low_quality": allow_low_quality,
        "bundles_read": bundles,
        "tasks_written": len(rows_out),
        "filtered_too_short": too_short,
        "filtered_duplicates": source_duplicates + task_duplicates,
        "filtered_source_duplicates": source_duplicates,
        "filtered_task_duplicates": task_duplicates,
        "filtered_not_verbatim_in_context": not_verbatim,
        "filtered_low_quality": low_quality,
        "filtered_not_hard_enough": not_hard,
        "clause_length_summary": summary,
        "hard_score_summary": hard_summary,
        "clause_key_counts": dict(clause_key_counts),
    }


def main() -> None:
    args = parse_args()
    train_dedupe_mode = args.train_dedupe_mode or args.dedupe_mode
    val_dedupe_mode = args.val_dedupe_mode or args.dedupe_mode
    stats = {
        "train": build_split(
            args.train_input,
            args.train_output,
            "train",
            args.min_clause_chars,
            args.max_train_tasks,
            args.instruction_mode,
            args.difficulty,
            args.min_hard_score,
            train_dedupe_mode,
            args.allow_train_low_quality,
        ),
        "val": build_split(
            args.val_input,
            args.val_output,
            "val",
            args.min_clause_chars,
            args.max_val_tasks,
            args.instruction_mode,
            args.difficulty,
            args.min_hard_score,
            val_dedupe_mode,
            args.allow_val_low_quality,
        ),
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
