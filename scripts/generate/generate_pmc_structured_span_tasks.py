#!/usr/bin/env python3
"""
Generate PMC anchored useful-span extraction tasks.

Design goals:
  - keep localization easy via subsection title + begins-with anchor
  - keep copying difficult via long, dense paragraph bundles
  - keep gold deterministic by extracting only contiguous paragraph spans

Example:
  python scripts/generate/generate_pmc_structured_span_tasks.py \
    --data-root data/pmc/extracted \
    --train-output generated_data/pmc_structured_span_tasks.train.jsonl \
    --val-output generated_data/pmc_structured_span_tasks.val.jsonl \
    --stats-output generated_data/pmc_structured_span_tasks.stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


ENTITY_PATTERNS = (
    re.compile(r"\byeast strains?\b", re.I),
    re.compile(r"\bstrains?(?:\s+and\s+plasmids?)?\b", re.I),
    re.compile(r"\bbacterial strains?\b", re.I),
    re.compile(r"\bcell lines?\b", re.I),
    re.compile(r"\bcell lines?\s+and\s+tissue samples?\b", re.I),
    re.compile(r"\bstudy population\b", re.I),
    re.compile(r"\bstudy participants\b", re.I),
    re.compile(r"\bparticipants\b", re.I),
    re.compile(r"\bpatients?\b", re.I),
    re.compile(r"\bsubjects?\b", re.I),
    re.compile(r"\bcohort\b", re.I),
    re.compile(r"\bstudy population\b", re.I),
    re.compile(r"\bstudy participants\b", re.I),
    re.compile(r"\bpatient and control dna samples?\b", re.I),
    re.compile(r"\bclinical material\b", re.I),
    re.compile(r"\bpopulation\b", re.I),
    re.compile(r"\bplant material\b", re.I),
)
STATS_PATTERNS = (
    re.compile(r"\bstatistic(?:al|s)?\b", re.I),
    re.compile(r"\bdata analysis\b", re.I),
    re.compile(r"\bpower analysis\b", re.I),
    re.compile(r"\bsimulation study\b", re.I),
)
PROTOCOL_PATTERNS = (
    re.compile(r"\bmethods?\b", re.I),
    re.compile(r"\bmethodology\b", re.I),
    re.compile(r"\bexperimental procedures?\b", re.I),
    re.compile(r"\bprotocols?\b", re.I),
    re.compile(r"\bcell culture\b", re.I),
    re.compile(r"\btissue culture\b", re.I),
    re.compile(r"\bcell cultures?\b", re.I),
    re.compile(r"\bculture conditions\b", re.I),
    re.compile(r"\bgrowth conditions\b", re.I),
    re.compile(r"\btransfection\b", re.I),
    re.compile(r"\btransfections\b", re.I),
    re.compile(r"\btransient transfections\b", re.I),
    re.compile(r"\bimmuno(?:blotting|fluorescence|histochemistry|cytochemistry)\b", re.I),
    re.compile(r"\bimmunoblot analysis\b", re.I),
    re.compile(r"\bimmunoprecipitation(?:s)?\b", re.I),
    re.compile(r"\bimmunostaining\b", re.I),
    re.compile(r"\bwestern blot(?:ting| analysis)?\b", re.I),
    re.compile(r"\bnorthern blot(?:ting| analysis)?\b", re.I),
    re.compile(r"\bsouthern blot(?:ting| analysis)?\b", re.I),
    re.compile(r"\bflow cytometry\b", re.I),
    re.compile(r"\bmicroscopy\b", re.I),
    re.compile(r"\bfluorescence microscopy\b", re.I),
    re.compile(r"\bconfocal microscopy\b", re.I),
    re.compile(r"\belectron microscopy\b", re.I),
    re.compile(r"\btransmission electron microscopy\b", re.I),
    re.compile(r"\blive cell imaging\b", re.I),
    re.compile(r"\bimage acquisition\b", re.I),
    re.compile(r"\bin situ hybridization\b", re.I),
    re.compile(r"\brt-pcr\b", re.I),
    re.compile(r"\breverse transcription-polymerase chain reaction\b", re.I),
    re.compile(r"\bquantitative real-time pcr\b", re.I),
    re.compile(r"\breal time pcr\b", re.I),
    re.compile(r"\bpcr\b", re.I),
    re.compile(r"\bplasmids?\b", re.I),
    re.compile(r"\bplasmid construction\b", re.I),
    re.compile(r"\bplasmid constructs?\b", re.I),
    re.compile(r"\bexpression constructs?\b", re.I),
    re.compile(r"\breagents?\b", re.I),
    re.compile(r"\bchemicals?\b", re.I),
    re.compile(r"\bantibodies\b", re.I),
    re.compile(r"\bmaterials?\b", re.I),
    re.compile(r"\bprotein purification\b", re.I),
    re.compile(r"\bkinase assays?\b", re.I),
    re.compile(r"\benzyme assays?\b", re.I),
    re.compile(r"\bproliferation assays?\b", re.I),
    re.compile(r"\bcell proliferation assay\b", re.I),
    re.compile(r"\bmtt assay\b", re.I),
    re.compile(r"\btunel assay\b", re.I),
    re.compile(r"\byeast two-hybrid assay\b", re.I),
    re.compile(r"\bin vitro binding assays?\b", re.I),
    re.compile(r"\bfermentation\b", re.I),
    re.compile(r"\birradiation\b", re.I),
    re.compile(r"\btreatments?\b", re.I),
    re.compile(r"\bsample preparation\b", re.I),
    re.compile(r"\bdata collection\b", re.I),
    re.compile(r"\bdata extraction\b", re.I),
    re.compile(r"\bdata processing\b", re.I),
    re.compile(r"\bstudy design\b", re.I),
    re.compile(r"\bsearch strategy\b", re.I),
    re.compile(r"\bstudy selection\b", re.I),
    re.compile(r"\bprocedures\b", re.I),
    re.compile(r"\bcell viability\b", re.I),
    re.compile(r"\bhistology\b", re.I),
    re.compile(r"\bprimers\b", re.I),
    re.compile(r"\bdna manipulations\b", re.I),
    re.compile(r"\bgenotyp(?:ing|e)\b", re.I),
    re.compile(r"\bconstructs?\b", re.I),
    re.compile(r"\bluciferase assays?\b", re.I),
    re.compile(r"\belectrophoretic mobility shift assays?(?:\s*\(emsa\))?\b", re.I),
    re.compile(r"\bmigration assays?\b", re.I),
    re.compile(r"\bcell migration assays?\b", re.I),
    re.compile(r"\badhesion assays?\b", re.I),
    re.compile(r"\bsequence analysis\b", re.I),
    re.compile(r"\bdna sequence analysis\b", re.I),
    re.compile(r"\bsequencing\b", re.I),
    re.compile(r"\bprotein analysis\b", re.I),
    re.compile(r"\bcell cycle analysis\b", re.I),
    re.compile(r"\belectrophysiological methods\b", re.I),
    re.compile(r"\bwestern blots?\b", re.I),
    re.compile(r"\bimmunoblots?\b", re.I),
    re.compile(r"\brt-pcr analysis\b", re.I),
    re.compile(r"\breal time pcr\b", re.I),
    re.compile(r"\bquantitative real-time pcr\b", re.I),
)
EXPANSION_FAMILY_PATTERNS = (
    re.compile(r"\bstudy design\b", re.I),
    re.compile(r"\bsearch strategy\b", re.I),
    re.compile(r"\bdata collection\b", re.I),
    re.compile(r"\bhistology\b", re.I),
    re.compile(r"\bcell viability\b", re.I),
    re.compile(r"\bluciferase assays?\b", re.I),
    re.compile(r"\bdata extraction\b", re.I),
    re.compile(r"\bsimulation study\b", re.I),
    re.compile(r"\belectrophoretic mobility shift assays?(?:\s*\(emsa\))?\b", re.I),
    re.compile(r"\bmigration assays?\b", re.I),
    re.compile(r"\badhesion assays?\b", re.I),
    re.compile(r"\bsequence analysis\b", re.I),
    re.compile(r"\bdna sequence analysis\b", re.I),
    re.compile(r"\bprotein analysis\b", re.I),
)
CONTEXT_OVERRIDE_TITLE_PATTERNS = (
    re.compile(r"\bstudy design\b", re.I),
    re.compile(r"\bsearch strategy\b", re.I),
    re.compile(r"\bdata collection\b", re.I),
)
LOW_QUALITY_OVERRIDE_TITLE_PATTERNS = (
    re.compile(r"\bstudy design\b", re.I),
    re.compile(r"\bsearch strategy\b", re.I),
    re.compile(r"\bdata collection\b", re.I),
)
MIN_WORDS_OVERRIDE_TITLE_PATTERNS = EXPANSION_FAMILY_PATTERNS
MULTI_BUNDLE_TITLE_PATTERNS = (
    re.compile(r"\bstudy design\b", re.I),
    re.compile(r"\bsearch strategy\b", re.I),
    re.compile(r"\bdata collection\b", re.I),
    re.compile(r"\bcell viability\b", re.I),
    re.compile(r"\bluciferase assays?\b", re.I),
)
METHODS_CONTEXT_PATTERNS = (
    re.compile(r"\bmethods?\b", re.I),
    re.compile(r"\bmaterials?\b", re.I),
    re.compile(r"\bmethodology\b", re.I),
    re.compile(r"\bexperimental procedures?\b", re.I),
    re.compile(r"\bstatistic(?:al|s)\b", re.I),
    re.compile(r"\bdata analysis\b", re.I),
    re.compile(r"\bstudy population\b", re.I),
)
BLOCKED_TITLE_PATTERNS = (
    re.compile(r"\bbackground\b", re.I),
    re.compile(r"\bresults?(?:\s+and\s+discussion)?\b", re.I),
    re.compile(r"\bdiscussion\b", re.I),
    re.compile(r"\bconclusions?\b", re.I),
    re.compile(r"\bsupplement(?:ary|al)?\b", re.I),
    re.compile(r"\bcompeting interests?\b", re.I),
    re.compile(r"\bauthors?' contributions?\b", re.I),
    re.compile(r"\bpre-publication history\b", re.I),
    re.compile(r"\bappendix\b", re.I),
    re.compile(r"\bquestionnaire\b", re.I),
    re.compile(r"\bchecklist\b", re.I),
    re.compile(r"\bconsort\b", re.I),
    re.compile(r"^step\s+\d+", re.I),
    re.compile(r"\brandomi[sz]ation\b", re.I),
    re.compile(r"\bsample size\b", re.I),
)
BLOCKED_STATS_TITLE_PATTERNS = (
    re.compile(r"^data analysis$", re.I),
    re.compile(r"^analysis of data$", re.I),
    re.compile(r"^power analysis$", re.I),
    re.compile(r"^normalization and statistical analysis$", re.I),
)
TOO_BROAD_TITLE_PATTERNS = (
    re.compile(r"^methods?$", re.I),
    re.compile(r"^materials?$", re.I),
    re.compile(r"^materials and methods$", re.I),
    re.compile(r"^general methods$", re.I),
    re.compile(r"^plasmids$", re.I),
    re.compile(r"^constructs$", re.I),
    re.compile(r"^reagents?$", re.I),
    re.compile(r"^chemicals?$", re.I),
    re.compile(r"^materials and reagents$", re.I),
    re.compile(r"^antibodies and reagents$", re.I),
    re.compile(r"^enzymes and chemicals$", re.I),
    re.compile(r"^chemicals and antibodies$", re.I),
    re.compile(r"^materials and plasmids$", re.I),
    re.compile(r"^endothelial cells and antibody reagents$", re.I),
    re.compile(r"^proteins and dna constructs$", re.I),
    re.compile(r"^chemicals and solutions$", re.I),
    re.compile(r"^constructs and virus$", re.I),
    re.compile(r"^animals?$", re.I),
    re.compile(r"^tissues?$", re.I),
)
LOW_QUALITY_TEXT_PATTERNS = (
    re.compile(r"\breferences?\b", re.I),
    re.compile(r"\btable\s+\d+\b", re.I),
    re.compile(r"\bfigure\s+\d+\b", re.I),
    re.compile(r"\badditional file\b", re.I),
    re.compile(r"\bclick here\b", re.I),
)
ENTITY_OVERRIDE_TO_PROTOCOL_PATTERNS = (
    re.compile(r"\bsample size\b", re.I),
    re.compile(r"\bsampling\b", re.I),
    re.compile(r"\btreat(?:ment|ments)\b", re.I),
    re.compile(r"\bprocessing\b", re.I),
    re.compile(r"\bpreparation\b", re.I),
    re.compile(r"\bisolation\b", re.I),
    re.compile(r"\bculture\b", re.I),
    re.compile(r"\btransfection\b", re.I),
    re.compile(r"\bmicroscopy\b", re.I),
    re.compile(r"\bblot(?:ting)?\b", re.I),
    re.compile(r"\bpcr\b", re.I),
    re.compile(r"\bpurification\b", re.I),
    re.compile(r"\bassay\b", re.I),
    re.compile(r"\banalysis\b", re.I),
    re.compile(r"\bsetup\b", re.I),
    re.compile(r"\brandomi[sz]ation\b", re.I),
    re.compile(r"\btransformation\b", re.I),
    re.compile(r"\bscreening\b", re.I),
    re.compile(r"\bsetting\b", re.I),
)
PROTOCOL_CUES = (
    re.compile(r"\b(?:incubat|treat|grew|grown|wash|washed|harvest|harvested|suspend|resuspend|lyse|fixed?|centrifug|irradiat)\w*\b", re.I),
    re.compile(r"\b(?:min|mins|minutes|h|hr|hrs|hours|days|°c|mM|μM|uM|mg|ml|μg|ug|Gy)\b", re.I),
)
ENTITY_CUES = (
    re.compile(r"\b(?:strain|genotype|wild type|mutant|cell line|patients?|subjects?|samples?)\b", re.I),
    re.compile(r"[A-Za-z0-9:+_-]+::[A-Za-z0-9:+_-]+"),
)
STATS_CUES = (
    re.compile(r"\bp\s*[<=>]\s*0?\.\d+", re.I),
    re.compile(r"\b(?:anova|t[- ]?test|mann[- ]?whitney|chi[- ]?square|confidence interval|regression|hazard ratio|odds ratio|wilcoxon|kruskal|friedman|bootstrap|software|sas|spss|stata|paired|unpaired|two[- ]sided|two sided|mean|means|median|standard deviation|standard error|s\\.d\\.|s\\.e\\.m\\.|sem)\b", re.I),
)
BAD_ANCHOR_TAIL_PATTERNS = (
    re.compile(r"\b(?:dr|mr|mrs|ms|prof|fig|figs|eq|eqs|ref|refs|no|nos|vs|ca|cf|cat|co|inc|ltd|corp)\.$", re.I),
    re.compile(r"\bet al\.$", re.I),
    re.compile(r"\b(?:dr|mr|mrs|ms|prof)\s+(?:[A-Z]\.){1,2}$", re.I),
)
MIXED_TITLE_SPLIT_RE = re.compile(r"\s*(?:,|/|;|\band\b)\s*", re.I)
BAD_INSTRUCTION_PATTERNS = (
    re.compile(r'\\"'),
)
BAD_GOLD_PATTERNS = (
    re.compile(r'[A-Za-z0-9]"[A-Za-z0-9]'),
    re.compile(r'\\(?![ntr"\\])'),
    re.compile(r"\bdata not shown\b", re.I),
    re.compile(r"\b(?:figure|fig\.|table)\s*\d", re.I),
    re.compile(
        r"\b(?:these results|our results|it was now found|it was reported|it was described|"
        r"suggest(?:ing)? that|indicat(?:e|es|ed|ing) that|demonstrat(?:e|es|ed|ing) that|"
        r"optimal dose was chosen because|in control experiments|as a control|"
        r"internal controls were always included|results from different batches were not averaged|"
        r"to validate the replication)\b",
        re.I,
    ),
)
BLOCKED_TASK_IDS = {
    "pmc_span.PMC1617214.015.00-00",
    "pmc_span.PMC239863.018.00-00",
    "pmc_span.PMC117243.018.00-00",
    "pmc_span.PMC331397.008.00-02",
    "pmc_span.PMC516040.020.01-01",
    "pmc_span.PMC516040.021.00-00",
    "pmc_span.PMC434497.012.00-01",
    "pmc_span.PMC65632.001.00-00",
    "pmc_span.PMC1097723.010.00-00",
    "pmc_span.PMC1559611.010.00-00",
    "pmc_span.PMC1224858.007.00-00",
    "pmc_span.PMC400736.003.00-00",
    "pmc_span.PMC1180468.004.00-00",
    "pmc_span.PMC35277.025.00-00",
    "pmc_span.PMC549047.012.00-00",
    "pmc_span.PMC305345.011.00-00",
    "pmc_span.PMC549561.003.00-00",
    "pmc_span.PMC1386658.008.00-00",
    "pmc_span.PMC1448181.011.00-00",
    "pmc_span.PMC1524963.010.00-00",
    "pmc_span.PMC1635735.011.00-00",
    "pmc_span.PMC420459.010.00-00",
    "pmc_span.PMC515295.009.00-00",
    "pmc_span.PMC524510.011.00-00",
    "pmc_span.PMC535351.008.00-00",
    "pmc_span.PMC122093.012.00-02",
    "pmc_span.PMC1479809.011.00-00",
    "pmc_span.PMC149371.007.01-01",
    "pmc_span.PMC1779629.007.00-01",
    "pmc_span.PMC420495.008.00-00",
    "pmc_span.PMC553973.014.00-00",
    "pmc_span.PMC1079803.013.00-00",
    "pmc_span.PMC1097723.014.00-00",
    "pmc_span.PMC1201157.008.00-00",
    "pmc_span.PMC122061.003.00-00",
    "pmc_span.PMC122061.005.00-00",
    "pmc_span.PMC122061.006.00-00",
    "pmc_span.PMC1450321.020.00-00",
    "pmc_span.PMC29067.006.00-00",
    "pmc_span.PMC385422.011.00-01",
    "pmc_span.PMC516025.016.00-00",
    "pmc_span.PMC548678.011.00-00",
    "pmc_span.PMC555569.010.00-00",
    "pmc_span.PMC1824786.002.00-00",
    "pmc_span.PMC270060.004.00-00",
    "pmc_span.PMC420458.014.00-00",
    "pmc_span.PMC481067.008.00-00",
    "pmc_span.PMC1298316.008.00-00",
    "pmc_span.PMC1702346.011.00-00",
    "pmc_span.PMC356908.012.00-00",
    "pmc_span.PMC1590014.011.00-00",
    "pmc_span.PMC544570.006.00-00",
    "pmc_span.PMC1180468.005.00-00",
    "pmc_span.PMC1570338.010.00-00",
    "pmc_span.PMC1781444.010.00-00",
    "pmc_span.PMC1832175.010.00-00",
}


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
    metadata: dict


@dataclass
class SectionRecord:
    index: int
    title: str
    depth: int
    paragraphs: list[str]
    full_text: str
    is_leaf: bool
    parent_index: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/pmc/extracted")
    parser.add_argument("--train-output", default="generated_data/pmc_structured_span_tasks.train.jsonl")
    parser.add_argument("--val-output", default="generated_data/pmc_structured_span_tasks.val.jsonl")
    parser.add_argument("--stats-output", default="generated_data/pmc_structured_span_tasks.stats.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-words", type=int, default=65)
    parser.add_argument("--max-words", type=int, default=900)
    parser.add_argument("--min-anchor-words", type=int, default=8)
    parser.add_argument("--max-anchor-words", type=int, default=24)
    parser.add_argument("--max-anchor-chars", type=int, default=320)
    parser.add_argument("--max-paragraphs", type=int, default=6)
    parser.add_argument("--min-context-words", type=int, default=500)
    parser.add_argument("--max-sibling-radius", type=int, default=3)
    parser.add_argument("--max-train-tasks", type=int, default=0)
    parser.add_argument("--max-val-tasks", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def choose_split(task_id: str, val_fraction: float) -> str:
    h = int(hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8], 16) % 1000
    return "val" if h < int(val_fraction * 1000) else "train"


def title_matches_any(title: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(p.search(title) for p in patterns)


def build_section_records(raw_sections: list[dict]) -> list[SectionRecord]:
    records: list[SectionRecord] = []
    stack: list[tuple[int, int]] = []
    for idx, sec in enumerate(raw_sections):
        depth = int(sec.get("depth", 0))
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent_index = stack[-1][1] if stack else None
        records.append(
            SectionRecord(
                index=idx,
                title=(sec.get("title") or "").strip(),
                depth=depth,
                paragraphs=[p.strip() for p in (sec.get("paragraphs") or []) if p and p.strip()],
                full_text=(sec.get("full_text") or "").strip(),
                is_leaf=bool(sec.get("is_leaf")),
                parent_index=parent_index,
            )
        )
        stack.append((depth, idx))
    return records


def is_blocked_title(title: str) -> bool:
    return any(p.search(title) for p in BLOCKED_TITLE_PATTERNS)


def is_mixed_title(title: str) -> bool:
    parts = [part.strip() for part in MIXED_TITLE_SPLIT_RE.split(title) if part.strip()]
    if len(parts) >= 3:
        return True
    has_entity = any(p.search(title) for p in ENTITY_PATTERNS)
    has_protocol = any(p.search(title) for p in PROTOCOL_PATTERNS)
    return has_entity and has_protocol and len(parts) >= 2


def has_multi_part_title(title: str) -> bool:
    parts = [part.strip() for part in MIXED_TITLE_SPLIT_RE.split(title) if part.strip()]
    return len(parts) >= 2


def classify_task_type(title: str) -> str | None:
    if is_blocked_title(title):
        return None
    if is_mixed_title(title):
        return None
    if any(p.search(title) for p in TOO_BROAD_TITLE_PATTERNS):
        return None
    if any(p.search(title) for p in BLOCKED_STATS_TITLE_PATTERNS):
        return None
    if any(p.search(title) for p in STATS_PATTERNS):
        return "statistical_method_extraction"
    if any(p.search(title) for p in ENTITY_OVERRIDE_TO_PROTOCOL_PATTERNS):
        if any(p.search(title) for p in PROTOCOL_PATTERNS):
            return "protocol_span_extraction"
    if any(p.search(title) for p in ENTITY_PATTERNS):
        return "entity_description_extraction"
    if any(p.search(title) for p in PROTOCOL_PATTERNS):
        return "protocol_span_extraction"
    return None


def ancestor_titles(records: list[SectionRecord], rec: SectionRecord) -> list[str]:
    titles: list[str] = []
    parent_idx = rec.parent_index
    while parent_idx is not None:
        parent = records[parent_idx]
        if parent.title:
            titles.append(parent.title)
        parent_idx = parent.parent_index
    return titles


def has_allowed_context(records: list[SectionRecord], rec: SectionRecord, task_type: str) -> bool:
    if title_matches_any(rec.title, CONTEXT_OVERRIDE_TITLE_PATTERNS):
        return True
    titles = [rec.title] + ancestor_titles(records, rec)
    ancestor_only = ancestor_titles(records, rec)
    if any(is_blocked_title(title) for title in ancestor_only):
        return False
    if task_type == "statistical_method_extraction":
        return any(
            any(p.search(title) for p in STATS_PATTERNS + METHODS_CONTEXT_PATTERNS)
            for title in titles
        )
    return any(any(p.search(title) for p in METHODS_CONTEXT_PATTERNS) for title in titles)


def paragraph_cue_score(paragraph: str, task_type: str) -> int:
    score = 0
    cues = PROTOCOL_CUES
    if task_type == "entity_description_extraction":
        cues = ENTITY_CUES
    elif task_type == "statistical_method_extraction":
        cues = STATS_CUES
    for pat in cues:
        if pat.search(paragraph):
            score += 1
    return score


def select_paragraph_bundle(
    paragraphs: list[str],
    task_type: str,
    min_words: int,
    max_words: int,
    max_paragraphs: int,
) -> tuple[int, int, str] | None:
    if not paragraphs:
        return None

    start_idx = 0
    best_score = -1
    for idx, para in enumerate(paragraphs[:3]):
        score = paragraph_cue_score(para, task_type)
        if score > best_score:
            best_score = score
            start_idx = idx

    selected: list[str] = []
    end_idx = start_idx
    for idx in range(start_idx, len(paragraphs)):
        para = paragraphs[idx].strip()
        if not para:
            continue
        candidate = selected + [para]
        candidate_text = "\n\n".join(candidate)
        if selected and word_count(candidate_text) > max_words:
            break
        selected.append(para)
        end_idx = idx
        if len(selected) >= max_paragraphs and word_count(candidate_text) >= min_words:
            break
        if word_count(candidate_text) >= min_words and idx + 1 < len(paragraphs):
            next_para = paragraphs[idx + 1].strip()
            if next_para and paragraph_cue_score(next_para, task_type) == 0:
                next_next_para = paragraphs[idx + 2].strip() if idx + 2 < len(paragraphs) else ""
                if not next_next_para or paragraph_cue_score(next_next_para, task_type) == 0:
                    break

    text = "\n\n".join(selected).strip()
    if not text:
        return None
    if word_count(text) < min_words:
        return None
    return start_idx, end_idx, text


def _grow_bundle_from_start(
    paragraphs: list[str],
    task_type: str,
    start_idx: int,
    min_words: int,
    max_words: int,
    max_paragraphs: int,
) -> tuple[int, int, str] | None:
    selected: list[str] = []
    end_idx = start_idx
    for idx in range(start_idx, len(paragraphs)):
        para = paragraphs[idx].strip()
        if not para:
            continue
        candidate = selected + [para]
        candidate_text = "\n\n".join(candidate)
        if selected and word_count(candidate_text) > max_words:
            break
        selected.append(para)
        end_idx = idx
        if len(selected) >= max_paragraphs and word_count(candidate_text) >= min_words:
            break
        if word_count(candidate_text) >= min_words and idx + 1 < len(paragraphs):
            next_para = paragraphs[idx + 1].strip()
            if next_para and paragraph_cue_score(next_para, task_type) == 0:
                next_next_para = paragraphs[idx + 2].strip() if idx + 2 < len(paragraphs) else ""
                if not next_next_para or paragraph_cue_score(next_next_para, task_type) == 0:
                    break

    text = "\n\n".join(selected).strip()
    if not text or word_count(text) < min_words:
        return None
    return start_idx, end_idx, text


def select_multiple_paragraph_bundles(
    paragraphs: list[str],
    task_type: str,
    min_words: int,
    max_words: int,
    max_paragraphs: int,
    max_bundles: int,
) -> list[tuple[int, int, str]]:
    if not paragraphs:
        return []

    candidates: list[tuple[int, int, str, int]] = []
    seen_spans: set[tuple[int, int]] = set()
    for start_idx, para in enumerate(paragraphs):
        if not para.strip():
            continue
        cue = paragraph_cue_score(para, task_type)
        if start_idx > 0 and cue == 0:
            continue
        bundle = _grow_bundle_from_start(
            paragraphs,
            task_type=task_type,
            start_idx=start_idx,
            min_words=min_words,
            max_words=max_words,
            max_paragraphs=max_paragraphs,
        )
        if bundle is None:
            continue
        s_idx, e_idx, text = bundle
        if (s_idx, e_idx) in seen_spans:
            continue
        seen_spans.add((s_idx, e_idx))
        cue_sum = sum(paragraph_cue_score(paragraphs[i], task_type) for i in range(s_idx, e_idx + 1))
        score = cue_sum * 1000 + word_count(text)
        candidates.append((s_idx, e_idx, text, score))

    candidates.sort(key=lambda item: (-item[3], item[0], item[1]))
    kept: list[tuple[int, int, str]] = []
    occupied: set[int] = set()
    for s_idx, e_idx, text, _score in candidates:
        span_indices = set(range(s_idx, e_idx + 1))
        if span_indices & occupied:
            continue
        kept.append((s_idx, e_idx, text))
        occupied |= span_indices
        if len(kept) >= max_bundles:
            break
    return kept


def anchor_from_text(text: str, min_words: int, max_words: int, max_chars: int) -> str | None:
    words = text.split()
    if len(words) < min_words:
        return None

    sentence_breaks = list(re.finditer(r"(?<=[.!?])\s+", text))
    for match in sentence_breaks:
        candidate = text[: match.start()].strip()
        if (
            candidate
            and len(candidate) <= max_chars
            and word_count(candidate) >= min_words
            and not any(p.search(candidate) for p in BAD_ANCHOR_TAIL_PATTERNS)
        ):
            return candidate

    first_line = normalize_ws(text.split("\n", 1)[0])
    if (
        first_line.endswith((".", "!", "?"))
        and len(first_line) <= max_chars
        and word_count(first_line) >= min_words
        and not any(p.search(first_line) for p in BAD_ANCHOR_TAIL_PATTERNS)
    ):
        return first_line
    return None


def sibling_leaf_indices(records: list[SectionRecord], target_idx: int) -> tuple[int | None, int | None]:
    target = records[target_idx]
    same_group = [
        r.index
        for r in records
        if r.is_leaf and r.depth == target.depth and r.parent_index == target.parent_index
    ]
    same_group.sort()
    pos = same_group.index(target_idx)
    prev_idx = same_group[pos - 1] if pos > 0 else None
    next_idx = same_group[pos + 1] if pos + 1 < len(same_group) else None
    return prev_idx, next_idx


def sibling_leaf_group(records: list[SectionRecord], target_idx: int) -> list[int]:
    target = records[target_idx]
    same_group = [
        r.index
        for r in records
        if r.is_leaf and r.depth == target.depth and r.parent_index == target.parent_index
    ]
    same_group.sort()
    return same_group


def heading_prefix(depth: int) -> str:
    return "#" * min(depth + 2, 6)


def _render_context_from_indices(paper_title: str, records: list[SectionRecord], target_idx: int, included: list[int]) -> str:
    target = records[target_idx]
    parts = [f"# {paper_title}"]
    if target.parent_index is not None:
        parent = records[target.parent_index]
        parts.append(f"{heading_prefix(parent.depth)} {parent.title}")
    for idx in included:
        sec = records[idx]
        parts.append(f"{heading_prefix(sec.depth + 1)} {sec.title}")
        parts.append(sec.full_text)
    return "\n\n".join(part for part in parts if part.strip()).strip()


def build_context(
    paper_title: str,
    records: list[SectionRecord],
    target_idx: int,
    min_context_words: int,
    max_sibling_radius: int,
) -> str:
    sibling_group = sibling_leaf_group(records, target_idx)
    pos = sibling_group.index(target_idx)

    def indices_for_radius(radius: int) -> list[int]:
        lo = max(0, pos - radius)
        hi = min(len(sibling_group), pos + radius + 1)
        return sibling_group[lo:hi]

    best_context = _render_context_from_indices(paper_title, records, target_idx, indices_for_radius(1))
    if word_count(best_context) >= min_context_words:
        return best_context

    for radius in range(2, max_sibling_radius + 1):
        candidate = _render_context_from_indices(paper_title, records, target_idx, indices_for_radius(radius))
        best_context = candidate
        if word_count(candidate) >= min_context_words:
            return candidate

    # If the local window is still short, include all sibling leaf subsections under the same parent.
    full_group_context = _render_context_from_indices(paper_title, records, target_idx, sibling_group)
    return full_group_context if word_count(full_group_context) > word_count(best_context) else best_context


def looks_low_quality(text: str) -> bool:
    if not text.strip():
        return True
    return any(p.search(text) for p in LOW_QUALITY_TEXT_PATTERNS)


def title_allows_low_quality_override(title: str) -> bool:
    return title_matches_any(title, LOW_QUALITY_OVERRIDE_TITLE_PATTERNS)


def title_min_words(title: str, default_min_words: int) -> int:
    if title_matches_any(title, MIN_WORDS_OVERRIDE_TITLE_PATTERNS):
        return min(default_min_words, 60)
    return default_min_words


def title_max_bundles(title: str) -> int:
    if title_matches_any(title, MULTI_BUNDLE_TITLE_PATTERNS):
        return 2
    return 1


def task_quality_issues(instruction: str, gold_text: str) -> list[str]:
    issues: list[str] = []
    for pat in BAD_INSTRUCTION_PATTERNS:
        if pat.search(instruction):
            issues.append("bad_instruction")
            break
    for pat in BAD_GOLD_PATTERNS:
        if pat.search(gold_text):
            issues.append("bad_gold")
    return issues


def infer_request_phrase(task_type: str, title: str, gold_text: str) -> str:
    title_l = title.lower()
    if has_multi_part_title(title):
        if task_type == "entity_description_extraction":
            return "the descriptive block in that subsection"
        if task_type == "statistical_method_extraction":
            return "the statistical analysis procedure in that subsection"
        return "the experimental procedure block in that subsection"
    if task_type == "entity_description_extraction":
        if "yeast" in title_l or "strain" in title_l:
            return "the strains used in the study"
        if "patient" in title_l or "subject" in title_l or "participant" in title_l or "cohort" in title_l:
            return "the study participants or cohort"
        if "population" in title_l:
            return "the study population"
        if "dna sample" in title_l or "tissue sample" in title_l or "clinical material" in title_l:
            return "the study samples or clinical material"
        if "plant material" in title_l:
            return "the biological material used in the study"
        if "cell line" in title_l:
            return "the cell lines used in the study"
        return "the biological materials or study entities"
    if task_type == "statistical_method_extraction":
        if "simulation study" in title_l:
            return "the simulation study procedure"
        if any(p.search(gold_text) for p in STATS_CUES):
            return "the statistical analysis procedure and any reported significance criteria"
        return "the statistical analysis procedure"
    if "data extraction" in title_l:
        return "the data extraction procedure"
    if "data processing" in title_l:
        return "the data processing procedure"
    if "migration assay" in title_l:
        return "the migration assay procedure"
    if "adhesion assay" in title_l:
        return "the adhesion assay procedure"
    if "sequence analysis" in title_l or "sequencing" in title_l or "genotyp" in title_l:
        return "the sequencing or genotyping procedure"
    if "protein analysis" in title_l:
        return "the protein analysis procedure"
    if "electrophoretic mobility shift assay" in title_l or "emsa" in title_l:
        return "the electrophoretic mobility shift assay procedure"
    if "culture" in title_l:
        return "culture conditions and related experimental handling"
    if "transfection" in title_l:
        return "the transfection procedure"
    if "western blot" in title_l or "immunoblot" in title_l:
        return "the immunoblot or western blot procedure"
    if "rt-pcr" in title_l or "pcr" in title_l:
        return "the PCR procedure"
    if "microscopy" in title_l:
        return "the microscopy procedure"
    if "search strategy" in title_l or "study selection" in title_l:
        return "how studies were identified or selected"
    if "data collection" in title_l or "study design" in title_l:
        return "the study design or data collection procedure"
    return "the experimental procedure described there"


def build_instruction(
    paper_title: str,
    section_title: str,
    task_type: str,
    anchor_text: str,
    gold_text: str,
    rng: random.Random,
) -> str:
    request = infer_request_phrase(task_type, section_title, gold_text)
    templates = [
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
    ]
    template = rng.choice(templates)
    return template.format(
        paper=paper_title.replace('"', '\\"'),
        section=section_title.replace('"', '\\"'),
        request=request,
        anchor=anchor_text.replace('"', '\\"'),
    )


def iter_tasks(data_root: Path, rng: random.Random, args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    title_counts = Counter()
    skipped = Counter()

    for sections_path in sorted(data_root.glob("PMC*/sections.json")):
        meta_path = sections_path.parent / "meta.json"
        if not meta_path.exists():
            skipped["missing_meta"] += 1
            continue
        meta = read_json(meta_path)
        paper_title = normalize_ws(str(meta.get("title") or sections_path.parent.name))
        records = build_section_records(read_json(sections_path))

        for rec in records:
            if not rec.is_leaf:
                continue
            if not rec.title or (
                looks_low_quality(rec.full_text) and not title_allows_low_quality_override(rec.title)
            ):
                skipped["low_quality_or_empty"] += 1
                continue
            task_type = classify_task_type(rec.title)
            if task_type is None:
                skipped["unmatched_title"] += 1
                continue
            if not has_allowed_context(records, rec, task_type):
                skipped["disallowed_context"] += 1
                continue
            min_words = title_min_words(rec.title, args.min_words)
            max_bundles = title_max_bundles(rec.title)
            if max_bundles > 1:
                bundles = select_multiple_paragraph_bundles(
                    rec.paragraphs,
                    task_type=task_type,
                    min_words=min_words,
                    max_words=args.max_words,
                    max_paragraphs=args.max_paragraphs,
                    max_bundles=max_bundles,
                )
            else:
                bundle = select_paragraph_bundle(
                    rec.paragraphs,
                    task_type=task_type,
                    min_words=min_words,
                    max_words=args.max_words,
                    max_paragraphs=args.max_paragraphs,
                )
                bundles = [bundle] if bundle is not None else []
            if not bundles:
                skipped["span_selection_failed"] += 1
                continue
            context = build_context(
                paper_title,
                records,
                rec.index,
                min_context_words=args.min_context_words,
                max_sibling_radius=args.max_sibling_radius,
            )
            for start_idx, end_idx, gold in bundles:
                if context.count(gold) != 1:
                    skipped["nonunique_gold_in_context"] += 1
                    continue
                anchor_text = anchor_from_text(
                    gold,
                    min_words=args.min_anchor_words,
                    max_words=args.max_anchor_words,
                    max_chars=args.max_anchor_chars,
                )
                if not anchor_text:
                    skipped["anchor_failed"] += 1
                    continue
                if context.count(anchor_text) != 1:
                    skipped["nonunique_anchor_in_context"] += 1
                    continue

                task_id = f"pmc_span.{sections_path.parent.name}.{rec.index:03d}.{start_idx:02d}-{end_idx:02d}"
                if task_id in BLOCKED_TASK_IDS:
                    skipped["blocked_task_id"] += 1
                    continue
                split = choose_split(task_id, args.val_fraction)
                instruction = build_instruction(paper_title, rec.title, task_type, anchor_text, gold, rng)
                quality_issues = task_quality_issues(instruction, gold)
                if quality_issues:
                    for issue in quality_issues:
                        skipped[issue] += 1
                    continue
                title_counts[rec.title] += 1
                tasks.append(
                    Task(
                        task_id=task_id,
                        case_id=f"{sections_path.parent.name}:{rec.title}:{start_idx}-{end_idx}",
                        task_type="copy_only",
                        span_type=task_type,
                        source_split=split,
                        source_dataset="pmc-structured-span",
                        instruction=instruction,
                        context=context,
                        gold_answer=gold,
                        gold_evidence=gold,
                        metadata={
                            "paper_id": sections_path.parent.name,
                            "paper_title": paper_title,
                            "section_title": rec.title,
                            "task_family": task_type,
                            "section_depth": rec.depth,
                            "parent_title": records[rec.parent_index].title if rec.parent_index is not None else "",
                            "start_paragraph_index": start_idx,
                            "end_paragraph_index": end_idx,
                            "paragraph_count": end_idx - start_idx + 1,
                            "anchor_text": anchor_text,
                            "context_chars": len(context),
                            "answer_chars": len(gold),
                            "answer_words": word_count(gold),
                        },
                    )
                )

    tasks.sort(key=lambda t: t.task_id)
    split_limits = {
        "train": args.max_train_tasks if args.max_train_tasks > 0 else None,
        "val": args.max_val_tasks if args.max_val_tasks > 0 else None,
    }
    final_tasks: list[Task] = []
    kept_counts = Counter()
    for task in tasks:
        limit = split_limits[task.source_split]
        if limit is not None and kept_counts[task.source_split] >= limit:
            continue
        final_tasks.append(task)
        kept_counts[task.source_split] += 1

    iter_tasks.skipped = skipped  # type: ignore[attr-defined]
    iter_tasks.title_counts = title_counts  # type: ignore[attr-defined]
    return final_tasks


def write_jsonl(path: Path, rows: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tasks = iter_tasks(Path(args.data_root), rng, args)
    train_tasks = [t for t in tasks if t.source_split == "train"]
    val_tasks = [t for t in tasks if t.source_split == "val"]

    write_jsonl(Path(args.train_output), train_tasks)
    write_jsonl(Path(args.val_output), val_tasks)

    skipped = getattr(iter_tasks, "skipped", Counter())
    title_counts = getattr(iter_tasks, "title_counts", Counter())
    stats = {
        "total_tasks": len(tasks),
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks),
        "task_family_counts": dict(Counter(t.span_type for t in tasks)),
        "top_section_titles": dict(title_counts.most_common(30)),
        "avg_answer_words": (sum(t.metadata["answer_words"] for t in tasks) / len(tasks)) if tasks else 0.0,
        "skipped_counts": dict(skipped),
    }
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
