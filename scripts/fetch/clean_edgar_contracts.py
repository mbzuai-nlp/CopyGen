#!/usr/bin/env python3
"""
Keep only EX-10 contract exhibits in an EDGAR download directory and
materialize plain-text versions for easier downstream use.

Example:
  python scripts/fetch/clean_edgar_contracts.py \
    --edgar-dir data/legal/edgar \
    --text-dir data/legal/edgar_text \
    --delete-non-contract
"""

from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import json
import re
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_contract_record(rec: dict) -> bool:
    exhibit_type = rec.get("exhibit_type", "").upper().strip()
    return exhibit_type.startswith("EX-10") and not exhibit_type.startswith("EX-101")


def normalize_text(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    lines: list[str] = []
    prev_blank = False
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t\r\f\v\xa0]+", " ", raw_line).strip()
        if not line:
            if not prev_blank and lines:
                lines.append("")
            prev_blank = True
            continue
        lines.append(line)
        prev_blank = False
    return "\n".join(lines).strip() + "\n"


class SimpleHTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "p",
        "div",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "section",
        "article",
        "center",
        "title",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def _append(self, text: str) -> None:
        if not text:
            return
        self.parts.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if tag == "br":
            self._append("\n")
        elif tag in {"td", "th"}:
            self._append("\t")
        elif tag == "hr":
            self._append("\n\n[[PAGEBREAK]]\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if tag in self.BLOCK_TAGS:
            self._append("\n")
        elif tag in {"td", "th"}:
            self._append("\t")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        collapsed = re.sub(r"\s+", " ", data)
        if collapsed.strip():
            # Preserve separation between inline styled fragments; spacing is
            # normalized later in post-processing.
            self._append(" " + collapsed.strip() + " ")

    def get_text(self) -> str:
        return "".join(self.parts)


def postprocess_extracted_text(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    text = re.sub(
        r"\[(?:Remainder of page intentionally left blank;\s*)?Signatures? appear on following page(?:\(s\))?\.?\s*(?:-\d+-)?\]",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"#\d{6,}(?:v\d+)?", " ", text)
    text = text.replace("\t", " ")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:?!\)])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r'([“‘"])\s+', r"\1", text)
    text = re.sub(r"\s+([”’\"])", r"\1", text)
    text = re.sub(r"(\d+)\.\s*([A-Z])", r"\1. \2", text)
    text = re.sub(r"(\([a-z]\))\s*([A-Z])", r"\1 \2", text)
    text = re.sub(r"\bRS Us\b", "RSUs", text)

    lines = [line.strip() for line in text.splitlines()]
    filtered: list[str] = []
    for line in lines:
        if not line:
            filtered.append("")
            continue
        if line == "[[PAGEBREAK]]":
            filtered.append("")
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if line == "Document":
            continue
        if re.fullmatch(
            r"\[?(?:remainder of page intentionally left blank;\s*)?signatures? appear on following page(?:\(s\))?\.?\s*(?:-\d+-)?\]?",
            line,
            flags=re.I,
        ):
            continue
        if re.fullmatch(r"#\d{6,}(?:v\d+)?", line):
            continue
        filtered.append(line)

    out: list[str] = []
    for line in filtered:
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if out and out[-1] and should_join_lines(out[-1], line):
            out[-1] = out[-1] + " " + line
        else:
            out.append(line)

    return normalize_text("\n".join(out))


def is_heading_like(line: str) -> bool:
    if len(line) <= 120 and line.upper() == line and any(ch.isalpha() for ch in line):
        return True
    if re.match(r"^\d+(\.\d+)*\.\s+[A-Z]", line):
        return True
    if re.match(r"^\([a-z]\)\s+", line):
        return True
    return False


def is_form_label_like(line: str) -> bool:
    if re.match(r"^[A-Z][A-Za-z0-9 /'&().-]{0,50}:\s*$", line):
        return True
    if re.match(r"^[A-Z][A-Za-z0-9 /'&().-]{0,50}:\s+.*$", line):
        return True
    return False


def should_join_lines(prev: str, curr: str) -> bool:
    if is_heading_like(prev) or is_heading_like(curr):
        return False
    if is_form_label_like(prev) or is_form_label_like(curr):
        return False
    if prev.endswith((".", ":", ";", "?", "!")):
        return False
    return True


def htmlish_to_text(raw: str) -> str:
    text = raw
    text_match = re.search(r"(?is)<TEXT>\s*(.*?)\s*</TEXT>", text)
    if text_match:
        text = text_match.group(1)

    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    parser = SimpleHTMLTextExtractor()
    parser.feed(text)
    parser.close()
    extracted = html.unescape(parser.get_text())
    return postprocess_extracted_text(extracted)


def plain_text_from_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?i)<html\b|<div\b|<TEXT>|<DOCUMENT>", raw):
        return htmlish_to_text(raw)
    return normalize_text(raw)


def prune_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only contract exhibits and write plain-text versions.")
    parser.add_argument("--edgar-dir", default="data/legal/edgar")
    parser.add_argument("--text-dir", default="data/legal/edgar_text")
    parser.add_argument("--delete-non-contract", action="store_true")
    args = parser.parse_args()

    edgar_dir = Path(args.edgar_dir)
    text_dir = Path(args.text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = edgar_dir / "manifest.jsonl"
    rows = load_manifest(manifest_path)
    contract_rows_raw = [row for row in rows if is_contract_record(row)]
    contract_rows: list[dict] = []
    seen_saved_paths: set[str] = set()
    for row in contract_rows_raw:
        saved_path_key = row.get("saved_path", "")
        if saved_path_key in seen_saved_paths:
            continue
        seen_saved_paths.add(saved_path_key)
        contract_rows.append(row)

    keep_rel_paths = set()
    for row in contract_rows:
        saved_path = Path(row["saved_path"])
        if saved_path.is_absolute():
            keep_rel_paths.add(saved_path.relative_to(edgar_dir))
        else:
            keep_rel_paths.add(saved_path.relative_to(edgar_dir))

    backup_manifest = edgar_dir / "manifest.all.jsonl"
    if rows and not backup_manifest.exists():
        backup_manifest.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in contract_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    text_manifest_path = text_dir / "manifest.jsonl"
    with text_manifest_path.open("w", encoding="utf-8") as text_manifest:
        for row in contract_rows:
            saved_path = Path(row["saved_path"])
            if not saved_path.is_absolute():
                saved_path = saved_path
            rel_path = saved_path.relative_to(edgar_dir)
            text_path = (text_dir / rel_path).with_suffix(".txt")
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(plain_text_from_file(saved_path), encoding="utf-8")

            meta = {
                **row,
                "source_path": str(saved_path),
                "text_path": str(text_path),
            }
            text_manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")

    deleted = 0
    if args.delete_non_contract:
        protected = {Path("manifest.jsonl"), Path("manifest.all.jsonl")}
        for path in sorted(p for p in edgar_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(edgar_dir)
            if rel in protected or rel in keep_rel_paths:
                continue
            path.unlink()
            deleted += 1
        prune_empty_dirs(edgar_dir)

    print(f"contracts_kept={len(contract_rows)}")
    print(f"text_written={len(contract_rows)}")
    print(f"deleted_non_contract_files={deleted}")
    print(f"text_dir={text_dir}")


if __name__ == "__main__":
    main()
