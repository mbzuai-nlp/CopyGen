#!/usr/bin/env python3
"""
Build contract clause-bundle extraction tasks from EDGAR exhibit text.

The task format is:
  - input context: one cleaned local Article window around the target clauses
  - output: a small JSON object containing 2-3 exact clause texts from that window

Example:
  conda run -n mad python scripts/generate/generate_legal_clause_tasks.py \
    --manifest data/legal/edgar_text/manifest.jsonl \
    --train-output data/legal_clause/legal_clause_bundle_tasks.train.jsonl \
    --val-output data/legal_clause/legal_clause_bundle_tasks.val.jsonl \
    --stats-output data/legal_clause/legal_clause_bundle_tasks.stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path


ARTICLE_RE = re.compile(r"^\s*ARTICLE\s+([IVXLC0-9]+)\b(?:\s*[-.:]?\s*(.*))?$", re.IGNORECASE)
SECTION_RE = re.compile(
    r"^\s*(?:Section\s+)?(?P<num>\d+(?:\.\d+)+)\.?\s+(?P<title>[^\n]+)$",
    re.IGNORECASE,
)
PAGE_RE = re.compile(r"^\s*[-–—]?\s*[A-Z]?\s*-?\d+\s*-?[A-Z]?\s*$")
MICRODOT_RE = re.compile(r"#ᴅʟᴘ_MICRODOT.*?_END", re.IGNORECASE)
EMBEDDED_SECTION_HEADING_RE = re.compile(
    r"(?i)(?:^|[.;])\s*(?:section\s+)?(?P<num>\d+(?:\.\d+)+)\.\s+"
    r"(?P<title>[A-Z][A-Za-z0-9/&',;:()\- ]{1,120}\.)\s*(?=(?:\([a-z]\)|[A-Z]))"
)
ARTICLE_TITLE_CUT_RE = re.compile(r"(?i)\b(?:section\s+)?\d+(?:\.\d+)+\.\s+")
ARTICLE_NOISE_RE = re.compile(r"(?i)\btable of contents\b")
SIGNATURE_STOP_RE = re.compile(
    r"(?i)^\s*(?:\(?signature page follows\)?|in witness whereof|witness the execution hereof|signature page|exhibit [a-z0-9]+|annex [a-z0-9]+|appendix [a-z0-9]+|schedule [a-z0-9]+|closing index)\b"
)
RUNNING_HEADER_SKIP_RE = re.compile(
    r"(?i)^\s*(?:execution version|[a-z0-9 ,.'()/:-]{0,80}agreement(?: \(\d{4}\))?\s*[-–—]?\s*page\s+\d+|page\s+\d+)\s*$"
)
INLINE_NOISE_RE = re.compile(
    r"\|US-DOCS\\\\[^|]+\|\||\bExecution Version\b|\bPage \d+\b|\b\d+\s*-\s*\d+\b|\b\d{6,}_\d+\b"
)
INLINE_SIGNATURE_STOP_RE = re.compile(
    r"(?i)\b(?:\(?signature page follows\)?|in witness whereof|witness the execution hereof|exhibit [a-z0-9]+|annex [a-z0-9]+|appendix [a-z0-9]+|closing index)\b"
)

BAD_DOC_HINTS = (
    "deferred",
    "bonus",
    "award",
    "compensation plan",
    "stock option",
    "rsu",
    "psu",
    "severance",
)
GOOD_DOC_HINTS = (
    "agreement",
    "amendment",
    "credit",
    "indenture",
    "facility",
    "guarant",
    "loan",
    "note",
)

CLAUSE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("confidentiality_clause", ("confidentiality",)),
    ("governing_law_clause", ("governing law", "choice of law")),
    ("consent_to_jurisdiction_clause", ("consent to jurisdiction", "submission to jurisdiction")),
    ("waiver_of_jury_trial_clause", ("waiver of jury trial", "jury trial waiver")),
    ("service_of_process_clause", ("service of process",)),
    ("notices_clause", ("notices",)),
    ("severability_clause", ("severability",)),
    ("counterparts_clause", ("counterparts",)),
    ("entire_agreement_clause", ("entire agreement", "integration")),
    ("amendments_and_waivers_clause", ("amendments and waivers", "amendments or waivers", "amendment and waiver")),
    ("successors_and_assigns_clause", ("successors and assigns",)),
    ("waiver_of_venue_clause", ("waiver of venue",)),
    ("nonreliance_clause", ("nonreliance",)),
    ("expenses_clause", ("expenses",)),
]

KEY_TO_DISPLAY = {
    "confidentiality_clause": "Confidentiality",
    "governing_law_clause": "Governing Law",
    "consent_to_jurisdiction_clause": "Consent to Jurisdiction",
    "waiver_of_jury_trial_clause": "Waiver of Jury Trial",
    "service_of_process_clause": "Service of Process",
    "notices_clause": "Notices",
    "severability_clause": "Severability",
    "counterparts_clause": "Counterparts",
    "entire_agreement_clause": "Entire Agreement",
    "amendments_and_waivers_clause": "Amendments and Waivers",
    "successors_and_assigns_clause": "Successors and Assigns",
    "waiver_of_venue_clause": "Waiver of Venue",
    "nonreliance_clause": "Nonreliance",
    "expenses_clause": "Expenses",
}

ENUM_PATTERN = re.compile(r"\(([a-z]|[ivx]+|[A-Z])\)")
XREF_PATTERN = re.compile(r"\b(?:Section|Sections|Article|Articles)\s+[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*", re.IGNORECASE)
ALLCAPS_WORD_PATTERN = re.compile(r"\b[A-Z][A-Z0-9/-]{2,}\b")
PROVISO_PATTERN = re.compile(r"\b(?:provided that|notwithstanding|except that|subject to)\b", re.IGNORECASE)
DIGIT_TOKEN_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:%|x|bps|days?|months?|years?)?\b")
QUOTE_PATTERN = re.compile(r"[\"'“”‘’]")


@dataclass(frozen=True)
class Section:
    number: str
    title: str
    heading: str
    text: str
    key: str | None


@dataclass(frozen=True)
class Article:
    label: str
    title: str
    heading: str
    text: str
    sections: list[Section]


@dataclass(frozen=True)
class ClauseFragility:
    score: int
    features: dict[str, int | bool]


@dataclass(frozen=True)
class BundleProfile:
    total_fragility: int
    min_fragility: int
    max_fragility: int
    total_clause_chars: int
    span_sections: int
    max_index_gap: int
    section_scores: list[int]
    section_features: list[dict[str, int | bool]]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_layout_line(raw_line: str) -> str:
    raw_line = raw_line.replace("\xa0", " ").rstrip()
    if not raw_line.strip():
        return ""
    indent = raw_line[: len(raw_line) - len(raw_line.lstrip(" "))]
    content = raw_line.strip()
    content = INLINE_NOISE_RE.sub("", content).strip()
    if not content:
        return ""
    return f"{indent}{content}" if indent else content


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def roman_to_int(s: str) -> int | None:
    s = s.upper().strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    prev = 0
    try:
        for ch in reversed(s):
            val = values[ch]
            if val < prev:
                total -= val
            else:
                total += val
                prev = val
    except KeyError:
        return None
    return total


def looks_like_contract(meta: dict) -> bool:
    hay = " ".join(
        [
            str(meta.get("exhibit_filename", "")),
            str(meta.get("exhibit_description", "")),
            str(meta.get("form", "")),
        ]
    ).lower()
    if any(bad in hay for bad in BAD_DOC_HINTS):
        return False
    return True


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = MICRODOT_RE.sub("", text)
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if PAGE_RE.fullmatch(line.strip()):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_article_title(title: str) -> str:
    title = normalize_spaces(title)
    if not title:
        return ""
    title = ARTICLE_TITLE_CUT_RE.split(title, maxsplit=1)[0].strip(" -.:;")
    if ARTICLE_NOISE_RE.search(title):
        return ""
    if len(title) > 140:
        return ""
    return title


def find_body_start(lines: list[str]) -> int:
    toc_idx = next((i for i, line in enumerate(lines) if "table of contents" in line.lower()), -1)
    article_positions: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        match = ARTICLE_RE.match(line.strip())
        if not match:
            continue
        label = match.group(1).upper()
        article_positions.setdefault(label, []).append(i)

    if toc_idx >= 0:
        repeated = []
        for label, positions in article_positions.items():
            if len(positions) >= 2 and positions[0] >= toc_idx:
                repeated.append(positions[1])
            elif len(positions) >= 2 and positions[0] < toc_idx < positions[1]:
                repeated.append(positions[1])
        if repeated:
            return min(repeated)

    first_article = next((i for i, line in enumerate(lines) if ARTICLE_RE.match(line.strip())), 0)
    return first_article


def split_articles(text: str, preserve_clause_layout: bool = False) -> list[Article]:
    lines = text.splitlines()
    if not lines:
        return []

    body_start = find_body_start(lines)
    lines = lines[body_start:]

    section_idxs = [i for i, line in enumerate(lines) if SECTION_RE.match(normalize_spaces(line))]
    if len(section_idxs) < 2:
        return []

    parsed_sections: list[tuple[int, Section]] = []
    for pos, start in enumerate(section_idxs):
        end = section_idxs[pos + 1] if pos + 1 < len(section_idxs) else len(lines)
        section = parse_section(lines[start:end], preserve_clause_layout=preserve_clause_layout)
        if section is not None:
            parsed_sections.append((start, section))
    if len(parsed_sections) < 2:
        return []

    articles: list[Article] = []
    current_major: str | None = None
    current_sections: list[Section] = []
    current_first_idx = 0

    def finalize_group(first_idx: int, sections: list[Section]) -> None:
        if len(sections) < 2:
            return
        article_label, article_title = find_article_header(lines, first_idx)
        if article_label is None:
            article_label = sections[0].number.split(".")[0]
        article_title = sanitize_article_title(article_title)
        article_heading = f"ARTICLE {article_label}"
        if article_title:
            article_heading += "\n" + article_title
        article_text = article_heading + "\n\n" + "\n\n".join(section.text for section in sections)
        articles.append(
            Article(
                label=article_label,
                title=article_title,
                heading=article_heading.strip(),
                text=article_text.strip(),
                sections=list(sections),
            )
        )

    for start_idx, section in parsed_sections:
        major = section.number.split(".")[0]
        if current_major is None:
            current_major = major
            current_sections = [section]
            current_first_idx = start_idx
            continue
        if major == current_major:
            current_sections.append(section)
            continue
        finalize_group(current_first_idx, current_sections)
        current_major = major
        current_sections = [section]
        current_first_idx = start_idx

    finalize_group(current_first_idx, current_sections)
    return articles


def parse_article(lines: list[str], preserve_clause_layout: bool = False) -> Article | None:
    if not lines:
        return None

    head_match = ARTICLE_RE.match(lines[0].strip())
    if not head_match:
        return None
    label = head_match.group(1).upper()
    title_parts: list[str] = []
    section_starts: list[int] = []
    for i in range(1, len(lines)):
        if SECTION_RE.match(lines[i].strip()):
            section_starts.append(i)
            break
        norm = normalize_spaces(lines[i])
        if norm:
            title_parts.append(norm)
    if not section_starts:
        return None

    first_section_idx = section_starts[0]
    for i in range(first_section_idx + 1, len(lines)):
        if SECTION_RE.match(lines[i].strip()):
            section_starts.append(i)

    sections: list[Section] = []
    for pos, start in enumerate(section_starts):
        end = section_starts[pos + 1] if pos + 1 < len(section_starts) else len(lines)
        section = parse_section(lines[start:end], preserve_clause_layout=preserve_clause_layout)
        if section is not None:
            sections.append(section)

    if len(sections) < 2:
        return None

    article_heading = f"ARTICLE {label}"
    if title_parts:
        article_heading += "\n" + normalize_spaces(" ".join(title_parts))

    rendered_sections = [section.text for section in sections]
    article_text = article_heading + "\n\n" + "\n\n".join(rendered_sections)
    return Article(
        label=label,
        title=normalize_spaces(" ".join(title_parts)),
        heading=article_heading.strip(),
        text=article_text.strip(),
        sections=sections,
    )


def find_article_header(lines: list[str], before_idx: int) -> tuple[str | None, str]:
    for i in range(before_idx, -1, -1):
        match = ARTICLE_RE.match(lines[i].strip())
        if not match:
            continue
        label = match.group(1).upper()
        title_bits: list[str] = []
        for j in range(i + 1, min(before_idx, i + 4)):
            line = normalize_spaces(lines[j])
            if not line:
                continue
            if SECTION_RE.match(line) or ARTICLE_RE.match(line):
                break
            title_bits.append(line)
        return label, sanitize_article_title(" ".join(title_bits))
    return None, ""


def classify_clause(title: str) -> str | None:
    norm = normalize_spaces(title).lower().strip(" .;:")
    for key, phrases in CLAUSE_RULES:
        if any(phrase in norm for phrase in phrases):
            return key
    return None


def parse_section(lines: list[str], preserve_clause_layout: bool = False) -> Section | None:
    if not lines:
        return None
    heading_line = normalize_spaces(lines[0])
    match = SECTION_RE.match(heading_line)
    if not match:
        return None

    number = match.group("num")
    raw_rest = normalize_spaces(match.group("title"))
    title, inline_body = split_title_and_body(raw_rest)
    if len(title) < 3 or len(title) > 140:
        return None

    body_parts: list[str] = []
    body_lines: list[str] = []
    if inline_body:
        body_parts.append(inline_body)
        body_lines.append(inline_body)
    for raw_line in lines[1:]:
        norm = normalize_spaces(raw_line)
        if not norm:
            if preserve_clause_layout and body_lines and body_lines[-1] != "":
                body_lines.append("")
            continue
        if SIGNATURE_STOP_RE.match(norm):
            break
        if RUNNING_HEADER_SKIP_RE.match(norm) or PAGE_RE.fullmatch(norm):
            continue
        inline_stop = INLINE_SIGNATURE_STOP_RE.search(norm)
        if inline_stop:
            norm = norm[: inline_stop.start()].strip()
            if not norm:
                break
            body_parts.append(norm)
            if preserve_clause_layout:
                body_lines.append(norm)
            break
        raw_layout_line = sanitize_layout_line(raw_line)
        norm = INLINE_NOISE_RE.sub("", norm).strip()
        if not norm:
            continue
        body_parts.append(norm)
        if preserve_clause_layout:
            body_lines.append(raw_layout_line or norm)
    body_text = " ".join(part for part in body_parts if part).strip()
    if not body_text:
        return None

    heading = f"{number}. {title}"
    if preserve_clause_layout:
        while body_lines and body_lines[-1] == "":
            body_lines.pop()
        rendered_body = "\n".join(body_lines).strip()
        if not rendered_body:
            return None
        text = f"{heading}\n{rendered_body}".strip()
    else:
        text = f"{heading} {body_text}".strip()
        text = re.sub(r"\s+", " ", text)
    embedded = EMBEDDED_SECTION_HEADING_RE.search(text[len(heading) :])
    if embedded and embedded.group("num") != number:
        text = text[: len(heading) + embedded.start()].rstrip()
    return Section(
        number=number,
        title=title,
        heading=heading,
        text=text,
        key=classify_clause(title),
    )


def split_title_and_body(raw_rest: str) -> tuple[str, str]:
    # Many EDGAR text files flatten the heading and the first sentence onto one line:
    #   16.1. GOVERNING LAW. THIS AGREEMENT ...
    # We treat the first sentence-like chunk as the section title.
    idx = raw_rest.find(". ")
    if idx != -1:
        title = raw_rest[: idx + 1].strip()
        body = raw_rest[idx + 2 :].strip()
        return title, body
    return raw_rest.strip(), ""


def eligible_clause_indices(article: Article, min_clause_chars: int, max_clause_chars: int) -> list[int]:
    eligible: list[int] = []
    seen_keys: set[str] = set()
    for idx, section in enumerate(article.sections):
        if section.key is None:
            continue
        if section.key in seen_keys:
            continue
        if not (min_clause_chars <= len(section.text) <= max_clause_chars):
            continue
        eligible.append(idx)
        seen_keys.add(section.key)
    return eligible


def parse_bundle_sizes(spec: str) -> tuple[int, ...]:
    sizes: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise ValueError(f"bundle sizes must be positive, got {size}")
        if size not in sizes:
            sizes.append(size)
    if not sizes:
        raise ValueError("bundle sizes must not be empty")
    return tuple(sizes)


def list_candidate_bundles(
    article: Article,
    min_clause_chars: int,
    max_clause_chars: int,
    bundle_sizes: tuple[int, ...],
) -> list[list[int]]:
    eligible = eligible_clause_indices(article, min_clause_chars, max_clause_chars)
    if len(eligible) < 2:
        return []

    bundles: list[list[int]] = []
    for size in bundle_sizes:
        if len(eligible) < size:
            continue
        for combo in itertools.combinations(eligible, size):
            bundles.append(list(combo))
    return bundles


def build_context_window(article: Article, selected_idxs: list[int], pre_sections: int, post_sections: int) -> tuple[str, list[Section]]:
    start = max(0, min(selected_idxs) - pre_sections)
    end = min(len(article.sections), max(selected_idxs) + post_sections + 1)
    window_sections = article.sections[start:end]
    context = article.heading + "\n\n" + "\n\n".join(section.text for section in window_sections)
    return context.strip(), window_sections


def article_matches_sections(article: Article) -> bool:
    label_num = roman_to_int(article.label)
    if label_num is None:
        return False
    majors = {section.number.split(".")[0] for section in article.sections}
    return majors == {str(label_num)}


def row_has_noise(context: str, gold_answer: str) -> bool:
    payload = context + "\n" + gold_answer
    patterns = [
        re.compile(r"(?i)\bIN WITNESS WHEREOF\b"),
        re.compile(r"(?i)\bEXHIBIT [A-Z0-9]+\b"),
        re.compile(r"(?i)\bANNEX [A-Z0-9]+\b"),
        re.compile(r"(?i)\bAPPENDIX [A-Z0-9]+\b"),
        re.compile(r"(?i)\bPage \d+\b"),
        re.compile(r"(?i)\bExecution Version\b"),
        re.compile(r"(?i)\|US-DOCS\\\\[^|]+\|\|"),
        re.compile(r"(?i)\bClosing Index\b"),
    ]
    return any(p.search(payload) for p in patterns)


def bundle_has_embedded_section(bundle: list[Section]) -> bool:
    for section in bundle:
        prefix_len = len(section.heading)
        suffix = section.text[prefix_len:]
        match = EMBEDDED_SECTION_HEADING_RE.search(suffix)
        if match and match.group("num") != section.number:
            return True
    return False


def compute_clause_fragility(section: Section, article: Article) -> ClauseFragility:
    text = section.text
    answer_chars = len(text)
    enum_count = len(ENUM_PATTERN.findall(text))
    xref_count = len(XREF_PATTERN.findall(text))
    semicolon_count = text.count(";")
    allcaps_count = len(ALLCAPS_WORD_PATTERN.findall(text))
    proviso_count = len(PROVISO_PATTERN.findall(text))
    digit_count = len(DIGIT_TOKEN_PATTERN.findall(text))
    quote_count = len(QUOTE_PATTERN.findall(text))
    paren_count = text.count("(")
    has_article_heading = bool(article.title)
    hard_key_family = section.key in {
        "governing_law_clause",
        "consent_to_jurisdiction_clause",
        "waiver_of_jury_trial_clause",
        "service_of_process_clause",
        "notices_clause",
        "severability_clause",
        "counterparts_clause",
        "entire_agreement_clause",
        "amendments_and_waivers_clause",
        "successors_and_assigns_clause",
        "waiver_of_venue_clause",
        "expenses_clause",
    }

    score = 0
    if answer_chars >= 1500:
        score += 3
    elif answer_chars >= 1100:
        score += 2
    elif answer_chars >= 800:
        score += 1
    if enum_count >= 5:
        score += 2
    elif enum_count >= 2:
        score += 1
    if xref_count >= 3:
        score += 2
    elif xref_count >= 1:
        score += 1
    if semicolon_count >= 6:
        score += 2
    elif semicolon_count >= 3:
        score += 1
    if allcaps_count >= 12:
        score += 2
    elif allcaps_count >= 6:
        score += 1
    if proviso_count >= 2:
        score += 2
    elif proviso_count >= 1:
        score += 1
    if digit_count >= 6:
        score += 1
    if quote_count >= 4:
        score += 1
    if paren_count >= 10:
        score += 1
    if has_article_heading:
        score += 1
    if hard_key_family:
        score += 1

    return ClauseFragility(
        score=score,
        features={
            "fragility_score": score,
            "answer_chars": answer_chars,
            "enum_count": enum_count,
            "xref_count": xref_count,
            "semicolon_count": semicolon_count,
            "allcaps_count": allcaps_count,
            "proviso_count": proviso_count,
            "digit_count": digit_count,
            "quote_count": quote_count,
            "paren_count": paren_count,
            "has_article_heading": has_article_heading,
            "hard_key_family": hard_key_family,
        },
    )


def compute_bundle_profile(
    article: Article,
    bundle_idxs: list[int],
    clause_profiles: dict[int, ClauseFragility],
) -> BundleProfile:
    scores = [clause_profiles[idx].score for idx in bundle_idxs]
    features = [clause_profiles[idx].features for idx in bundle_idxs]
    total_clause_chars = sum(len(article.sections[idx].text) for idx in bundle_idxs)
    sorted_idxs = sorted(bundle_idxs)
    gaps = [b - a for a, b in zip(sorted_idxs, sorted_idxs[1:])]
    return BundleProfile(
        total_fragility=sum(scores),
        min_fragility=min(scores),
        max_fragility=max(scores),
        total_clause_chars=total_clause_chars,
        span_sections=sorted_idxs[-1] - sorted_idxs[0] + 1,
        max_index_gap=max(gaps, default=0),
        section_scores=list(scores),
        section_features=list(features),
    )


def order_bundle_for_output(bundle: list[Section], task_seed: str, shuffle_output_order: bool) -> list[Section]:
    ordered = list(bundle)
    if not shuffle_output_order or len(ordered) < 2:
        return ordered
    rng = random.Random(stable_int(task_seed))
    rng.shuffle(ordered)
    if ordered == bundle:
        ordered = ordered[1:] + ordered[:1]
    return ordered


def render_instruction(bundle: list[Section], preserve_clause_layout: bool = False) -> str:
    requested = [f'the clause headed "{section.title}"' for section in bundle if section.key]
    keys = [section.key for section in bundle if section.key]
    numbered = "\n".join(f"{i}. {name}" for i, name in enumerate(requested, start=1))
    key_list = ", ".join(f'"{key}"' for key in keys)
    preservation_note = "Include each section heading and preserve the original wording."
    if preserve_clause_layout:
        preservation_note = (
            "Include each section heading and preserve the original wording, line breaks, and indentation."
        )
    return (
        "Extract the full text of the following clauses exactly as written in the article below:\n"
        f"{numbered}\n\n"
        f"Return a JSON object with keys {key_list}. "
        f"{preservation_note} "
        "Output only the JSON object."
    )


def build_gold_answer(bundle: list[Section]) -> str:
    payload = {section.key: section.text for section in bundle if section.key}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def stable_split_id(s: str, val_ratio: float) -> str:
    score = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if score < val_ratio else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate legal clause bundle extraction tasks.")
    parser.add_argument("--manifest", default="data/legal/edgar_text/manifest.jsonl")
    parser.add_argument("--train-output", default="data/legal_clause/legal_clause_bundle_tasks.train.jsonl")
    parser.add_argument("--val-output", default="data/legal_clause/legal_clause_bundle_tasks.val.jsonl")
    parser.add_argument("--stats-output", default="data/legal_clause/legal_clause_bundle_tasks.stats.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.17)
    parser.add_argument("--min-clause-chars", type=int, default=250)
    parser.add_argument("--max-clause-chars", type=int, default=2200)
    parser.add_argument("--min-context-chars", type=int, default=900)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--window-pre-sections", type=int, default=1)
    parser.add_argument("--window-post-sections", type=int, default=1)
    parser.add_argument("--max-bundles-per-article", type=int, default=4)
    parser.add_argument("--bundle-sizes", default="3,2", help="Comma-separated bundle sizes to consider, e.g. '4,3,2'.")
    parser.add_argument("--ranking-mode", choices=["baseline", "fragile"], default="baseline")
    parser.add_argument("--min-clause-fragility", type=int, default=0)
    parser.add_argument("--shuffle-output-order", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preserve-clause-layout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-tasks", type=int, default=0)
    parser.add_argument("--max-val-tasks", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    bundle_sizes = parse_bundle_sizes(args.bundle_sizes)

    train_path = Path(args.train_output)
    val_path = Path(args.val_output)
    stats_path = Path(args.stats_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in Path(args.manifest).open("r", encoding="utf-8") if line.strip()]
    random.shuffle(rows)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    stats = Counter()
    clause_key_counter = Counter()

    for meta in rows:
        if not looks_like_contract(meta):
            stats["skipped_non_contract_like"] += 1
            continue

        text_path = Path(meta["text_path"])
        if not text_path.exists():
            stats["skipped_missing_text"] += 1
            continue

        cleaned = clean_text(text_path.read_text(encoding="utf-8", errors="ignore"))
        articles = split_articles(cleaned, preserve_clause_layout=args.preserve_clause_layout)
        if not articles:
            stats["skipped_no_articles"] += 1
            continue

        split = stable_split_id(f"{meta['accession']}::{meta['exhibit_filename']}", args.val_ratio)
        for article in articles:
            if not article_matches_sections(article):
                stats["skipped_article_mismatch"] += 1
                continue
            clause_profiles = {
                idx: compute_clause_fragility(section, article)
                for idx, section in enumerate(article.sections)
                if section.key is not None
            }
            candidate_bundles = list_candidate_bundles(
                article,
                args.min_clause_chars,
                args.max_clause_chars,
                bundle_sizes,
            )
            if not candidate_bundles:
                stats["skipped_bundle_too_small"] += 1
                continue

            ranked: list[tuple[tuple, list[int], str, list[Section], BundleProfile]] = []
            for bundle_idxs in candidate_bundles:
                context, window_sections = build_context_window(
                    article,
                    bundle_idxs,
                    args.window_pre_sections,
                    args.window_post_sections,
                )
                if not (args.min_context_chars <= len(context) <= args.max_context_chars):
                    stats["skipped_context_length"] += 1
                    continue
                profile = compute_bundle_profile(article, bundle_idxs, clause_profiles)
                if profile.min_fragility < args.min_clause_fragility:
                    stats["skipped_low_fragility"] += 1
                    continue
                if args.ranking_mode == "fragile":
                    rank_key = (
                        -profile.total_fragility,
                        -profile.min_fragility,
                        -profile.total_clause_chars,
                        -len(bundle_idxs),
                        -profile.max_index_gap,
                        -profile.span_sections,
                        len(context),
                        bundle_idxs,
                    )
                else:
                    rank_key = (-len(bundle_idxs), len(context), bundle_idxs)
                ranked.append((rank_key, bundle_idxs, context, window_sections, profile))

            if not ranked:
                continue

            ranked.sort(key=lambda x: x[0])
            kept_for_article = 0
            seen_key_sets: set[tuple[str, ...]] = set()
            for bundle_num, (_, bundle_idxs, context, window_sections, profile) in enumerate(ranked, start=1):
                if kept_for_article >= args.max_bundles_per_article:
                    break
                if split == "train" and args.max_train_tasks and len(train_rows) >= args.max_train_tasks:
                    break
                if split == "val" and args.max_val_tasks and len(val_rows) >= args.max_val_tasks:
                    break

                bundle_doc_order = [article.sections[idx] for idx in bundle_idxs]
                if bundle_has_embedded_section(bundle_doc_order):
                    stats["skipped_embedded_section"] += 1
                    continue
                selected_keys = tuple(section.key for section in bundle_doc_order if section.key)
                if selected_keys in seen_key_sets:
                    continue
                seen_key_sets.add(selected_keys)
                task_id = (
                    f"legal_clause.{split}.{meta['accession']}.{meta['exhibit_filename']}."
                    f"article_{article.label.lower()}.bundle_{kept_for_article + 1}"
                )
                bundle = order_bundle_for_output(
                    bundle_doc_order,
                    task_seed=task_id,
                    shuffle_output_order=args.shuffle_output_order,
                )
                selected_key_list = [section.key for section in bundle if section.key]
                gold_answer = build_gold_answer(bundle)
                if row_has_noise(context, gold_answer):
                    stats["skipped_noise"] += 1
                    continue
                row = {
                    "task_id": task_id,
                    "case_id": f"{meta['accession']}::{meta['exhibit_filename']}::ARTICLE_{article.label}::bundle_{kept_for_article + 1}",
                    "task_type": "copy_json_bundle",
                    "span_type": "legal_article_clause_bundle",
                    "source_split": split,
                    "source_dataset": "edgar_contracts",
                    "instruction": render_instruction(bundle, preserve_clause_layout=args.preserve_clause_layout),
                    "context": context,
                    "gold_answer": gold_answer,
                    "gold_evidence": "\n\n".join(section.text for section in bundle),
                    "metadata": {
                        "cik": meta["cik"],
                        "accession": meta["accession"],
                        "form": meta["form"],
                        "filing_date": meta["filing_date"],
                        "exhibit_type": meta["exhibit_type"],
                        "exhibit_filename": meta["exhibit_filename"],
                        "text_path": meta["text_path"],
                        "article_label": article.label,
                        "article_title": article.title,
                        "window_section_numbers": [section.number for section in window_sections],
                        "selected_clause_keys": selected_key_list,
                        "selected_clause_titles": [section.title for section in bundle],
                        "selected_clause_keys_doc_order": [section.key for section in bundle_doc_order if section.key],
                        "selected_clause_titles_doc_order": [section.title for section in bundle_doc_order],
                        "context_chars": len(context),
                        "answer_chars": len(gold_answer),
                        "ranking_mode": args.ranking_mode,
                        "shuffle_output_order": args.shuffle_output_order,
                        "preserve_clause_layout": args.preserve_clause_layout,
                        "bundle_total_fragility": profile.total_fragility,
                        "bundle_min_fragility": profile.min_fragility,
                        "bundle_max_fragility": profile.max_fragility,
                        "bundle_total_clause_chars": profile.total_clause_chars,
                        "bundle_span_sections": profile.span_sections,
                        "bundle_max_index_gap": profile.max_index_gap,
                        "selected_clause_fragility_scores_doc_order": profile.section_scores,
                        "selected_clause_fragility_features_doc_order": profile.section_features,
                    },
                }
                for key in selected_key_list:
                    clause_key_counter[key] += 1
                if split == "train":
                    train_rows.append(row)
                else:
                    val_rows.append(row)
                stats["kept_tasks"] += 1
                kept_for_article += 1

    with train_path.open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "stats": dict(stats),
        "clause_key_counts": dict(clause_key_counter),
    }
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"train_count={len(train_rows)}")
    print(f"val_count={len(val_rows)}")
    print(f"train_output={train_path}")
    print(f"val_output={val_path}")
    print(f"stats_output={stats_path}")


if __name__ == "__main__":
    main()
