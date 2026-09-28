#!/usr/bin/env python3
"""
Extract PMC article packages into raw NXML plus lightweight structured text.

Example:
  python scripts/fetch/extract_pmc_packages.py \
    --packages-dir data/pmc/packages \
    --out-dir data/pmc/extracted \
    --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

SKIP_BODY_TAGS = {
    "fig",
    "fig-group",
    "graphic",
    "media",
    "table-wrap",
    "table-wrap-group",
    "ref-list",
    "ack",
    "notes",
    "fn-group",
    "app-group",
    "supplementary-material",
}

PARA_LIKE_TAGS = {
    "p",
    "disp-quote",
    "statement",
    "boxed-text",
}


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def cleanup_inline_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:?!\)])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r'([“‘"]) +', r"\1", text)
    text = re.sub(r" +([”’\"])", r"\1", text)
    text = re.sub(
        r"\[\s*([0-9,\-– ]+)\s*\]",
        lambda m: "[" + re.sub(r"\s*-\s*", "-", re.sub(r"\s*,\s*", ", ", m.group(1).strip())) + "]",
        text,
    )
    return text.strip()


def elem_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    pieces = [piece.strip() for piece in elem.itertext() if piece and piece.strip()]
    return cleanup_inline_text(" ".join(pieces))


def first_child(elem: ET.Element | None, name: str) -> ET.Element | None:
    if elem is None:
        return None
    for child in elem:
        if local_name(child.tag) == name:
            return child
    return None


def all_children(elem: ET.Element | None, name: str) -> list[ET.Element]:
    if elem is None:
        return []
    return [child for child in elem if local_name(child.tag) == name]


def collect_abstract(front: ET.Element | None) -> str:
    article_meta = first_child(front, "article-meta")
    abstract = first_child(article_meta, "abstract")
    if abstract is None:
        return ""
    sec_children = [child for child in abstract if local_name(child.tag) == "sec"]
    if sec_children:
        blocks: list[str] = []
        for sec in sec_children:
            title = elem_text(first_child(sec, "title"))
            if title:
                blocks.append(title)
            paragraphs = gather_text_blocks(sec)
            if paragraphs:
                blocks.append("\n\n".join(paragraphs))
        if blocks:
            return "\n\n".join(blocks)
    parts: list[str] = []
    for child in abstract:
        if local_name(child.tag) in {"title"}:
            continue
        text = elem_text(child)
        if text:
            parts.append(text)
    if not parts:
        text = elem_text(abstract)
        return text
    return "\n\n".join(parts)


def gather_text_blocks(elem: ET.Element) -> list[str]:
    blocks: list[str] = []
    for child in elem:
        tag = local_name(child.tag)
        if tag in SKIP_BODY_TAGS or tag == "title":
            continue
        if tag in PARA_LIKE_TAGS:
            text = elem_text(child)
            if text:
                blocks.append(text)
            continue
        if tag == "list":
            for item in child.iter():
                if local_name(item.tag) == "list-item":
                    text = elem_text(item)
                    if text:
                        blocks.append(text)
            continue
        if tag == "sec":
            continue
        text = elem_text(child)
        if text:
            blocks.append(text)
    return blocks


def flatten_section_with_children(sec: ET.Element) -> list[str]:
    texts = gather_text_blocks(sec)
    for child in sec:
        if local_name(child.tag) == "sec":
            texts.extend(flatten_section_with_children(child))
    return texts


def section_to_markdown(sec: ET.Element, depth: int, sections: list[dict], out_lines: list[str]) -> None:
    title = elem_text(first_child(sec, "title")) or f"Untitled section {len(sections) + 1}"
    level = min(depth + 2, 6)
    out_lines.extend([f"{'#' * level} {title}", ""])

    direct_paragraphs = gather_text_blocks(sec)
    section_entry = {
        "title": title,
        "depth": depth,
        "paragraphs": direct_paragraphs,
        "full_text": "\n\n".join(flatten_section_with_children(sec)).strip(),
        "is_leaf": not any(local_name(child.tag) == "sec" for child in sec),
    }
    for text in direct_paragraphs:
        if text:
            out_lines.extend([text, ""])
    sections.append(section_entry)

    for child in sec:
        if local_name(child.tag) == "sec":
            section_to_markdown(child, depth + 1, sections, out_lines)


def package_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".tar.gz"):
        return name[:-7]
    return path.stem


def extract_one(package_path: Path, out_dir: Path) -> dict | None:
    pmcid = package_stem(package_path)
    target_dir = out_dir / pmcid
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(package_path, "r:gz") as tf:
        nxml_members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".nxml")]
        if not nxml_members:
            return None
        member = nxml_members[0]
        raw = tf.extractfile(member).read()

    xml_path = target_dir / "article.nxml"
    xml_path.write_bytes(raw)

    root = ET.fromstring(raw)
    front = first_child(root, "front")
    body = first_child(root, "body")

    article_meta = first_child(front, "article-meta")
    title_group = first_child(article_meta, "title-group")
    title = elem_text(first_child(title_group, "article-title"))
    if not title:
        title = elem_text(first_child(first_child(front, "journal-meta"), "journal-title")) or pmcid

    pmid = ""
    pmc_from_xml = ""
    if article_meta is not None:
        for child in article_meta:
            if local_name(child.tag) != "article-id":
                continue
            pub_type = child.attrib.get("pub-id-type", "")
            value = elem_text(child)
            if pub_type == "pmid":
                pmid = value
            if pub_type == "pmc":
                pmc_from_xml = value

    abstract = collect_abstract(front)
    journal_meta = first_child(front, "journal-meta")
    journal_title = elem_text(first_child(journal_meta, "journal-title"))

    markdown_lines = [f"# {title}", ""]
    if abstract:
        markdown_lines.extend(["## Abstract", "", abstract, ""])

    sections: list[dict] = []
    if body is not None:
        for child in body:
            tag = local_name(child.tag)
            if tag == "sec":
                section_to_markdown(child, 0, sections, markdown_lines)
            else:
                text = elem_text(child)
                if text:
                    markdown_lines.extend([text, ""])

    markdown_text = "\n".join(markdown_lines).strip() + "\n"
    (target_dir / "full_text.md").write_text(markdown_text, encoding="utf-8")
    (target_dir / "sections.json").write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "pmcid": pmc_from_xml or pmcid.removeprefix("PMC"),
        "pmcid_label": pmcid,
        "pmid": pmid,
        "title": title,
        "journal_title": journal_title,
        "package_path": str(package_path),
        "xml_path": str(xml_path),
        "markdown_path": str(target_dir / "full_text.md"),
        "sections_path": str(target_dir / "sections.json"),
        "num_sections": len(sections),
        "num_leaf_sections": sum(1 for section in sections if section["is_leaf"]),
    }
    (target_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PMC tarballs into NXML and structured text.")
    parser.add_argument("--packages-dir", default="data/pmc/packages")
    parser.add_argument("--out-dir", default="data/pmc/extracted")
    parser.add_argument("--limit", type=int, default=0, help="0 means all packages in the directory.")
    args = parser.parse_args()

    packages_dir = Path(args.packages_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    package_paths = sorted(packages_dir.glob("*.tar.gz"))
    if args.limit:
        package_paths = package_paths[: args.limit]

    manifest_path = out_dir / "manifest.jsonl"
    count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for package_path in package_paths:
            meta = extract_one(package_path, out_dir)
            if meta is None:
                continue
            manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")
            count += 1

    print(f"packages_processed={count}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
