#!/usr/bin/env python3
"""
Download SEC EDGAR exhibit attachments (for example EX-10 material contracts)
from modern electronic filings into a local directory.

Workflow:
  1. Fetch company submissions JSON from data.sec.gov.
  2. Select recent filings by form type.
  3. Fetch the raw filing .txt from Archives.
  4. Parse DOCUMENT blocks and download exhibit attachments whose TYPE matches
     a target prefix such as EX-10.

Example:
  python scripts/fetch/fetch_edgar_exhibits.py \
    --out-dir data/legal/edgar \
    --cik 320193 \
    --cik 789019 \
    --user-agent "CopyGen research contact you@example.com" \
    --limit-filings-per-cik 20 \
    --download-limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/"


@dataclass
class FilingRecord:
    cik: int
    accession: str
    form: str
    filing_date: str
    primary_document: str


@dataclass
class ExhibitRecord:
    cik: int
    accession: str
    form: str
    filing_date: str
    exhibit_type: str
    exhibit_description: str
    exhibit_filename: str
    filing_txt_url: str
    exhibit_url: str
    saved_path: str


def fetch_bytes(url: str, user_agent: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_json(url: str, user_agent: str) -> dict:
    return json.loads(fetch_bytes(url, user_agent).decode("utf-8"))


def fetch_text(url: str, user_agent: str) -> str:
    data = fetch_bytes(url, user_agent)
    return data.decode("utf-8", errors="replace")


def iter_ciks(args_ciks: list[str], cik_file: str | None) -> list[int]:
    vals: list[str] = list(args_ciks)
    if cik_file:
        vals.extend(
            line.strip()
            for line in Path(cik_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    out: list[int] = []
    for val in vals:
        digits = re.sub(r"\D", "", val)
        if not digits:
            continue
        out.append(int(digits))
    return out


def recent_filings(submissions: dict, allowed_forms: set[str], limit: int) -> list[FilingRecord]:
    recent = submissions.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    cik = int(submissions.get("cik"))

    kept: list[FilingRecord] = []
    for accession, form, filing_date, primary_doc in zip(accessions, forms, filing_dates, primary_docs):
        if allowed_forms and form not in allowed_forms:
            continue
        kept.append(
            FilingRecord(
                cik=cik,
                accession=accession,
                form=form,
                filing_date=filing_date,
                primary_document=primary_doc,
            )
        )
        if limit and len(kept) >= limit:
            break
    return kept


def filing_txt_url(rec: FilingRecord) -> str:
    accession_nodash = rec.accession.replace("-", "")
    return f"{ARCHIVES_BASE}{rec.cik}/{accession_nodash}/{rec.accession}.txt"


def archive_dir_url(rec: FilingRecord) -> str:
    accession_nodash = rec.accession.replace("-", "")
    return f"{ARCHIVES_BASE}{rec.cik}/{accession_nodash}/"


def parse_document_blocks(filing_txt: str) -> list[dict[str, str]]:
    blocks = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", filing_txt, flags=re.I | re.S)
    out: list[dict[str, str]] = []
    for block in blocks:
        type_match = re.search(r"<TYPE>\s*([^\n\r<]+)", block, flags=re.I)
        filename_match = re.search(r"<FILENAME>\s*([^\n\r<]+)", block, flags=re.I)
        desc_match = re.search(r"<DESCRIPTION>\s*([^\n\r<]+)", block, flags=re.I)
        text_match = re.search(r"<TEXT>\s*(.*)", block, flags=re.I | re.S)
        if not type_match or not filename_match:
            continue
        out.append(
            {
                "type": type_match.group(1).strip(),
                "filename": filename_match.group(1).strip(),
                "description": desc_match.group(1).strip() if desc_match else "",
                "text": text_match.group(1).strip() if text_match else "",
            }
        )
    return out


def is_target_exhibit(doc: dict[str, str], type_prefixes: tuple[str, ...]) -> bool:
    doc_type = doc.get("type", "").upper().strip()
    if not doc_type:
        return False
    if doc_type.startswith("EX-101"):
        return False
    return any(doc_type.startswith(prefix) for prefix in type_prefixes)


def safe_name(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EDGAR exhibit attachments into a local directory.")
    parser.add_argument("--out-dir", default="data/legal/edgar")
    parser.add_argument("--cik", action="append", default=[], help="CIK (repeatable).")
    parser.add_argument("--cik-file", default="", help="Optional file with one CIK per line.")
    parser.add_argument("--form", action="append", default=["8-K", "10-K", "10-Q", "S-1"])
    parser.add_argument("--type-prefix", action="append", default=["EX-10"])
    parser.add_argument("--limit-filings-per-cik", type=int, default=20)
    parser.add_argument("--download-limit", type=int, default=20)
    parser.add_argument("--delay-sec", type=float, default=0.2, help="Delay between SEC requests.")
    parser.add_argument("--user-agent", required=True, help="Required by SEC, e.g. 'CopyGen research you@example.com'")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    ciks = iter_ciks(args.cik, args.cik_file or None)
    if not ciks:
        raise SystemExit("No CIKs provided.")

    allowed_forms = {x.strip() for x in args.form if x.strip()}
    type_prefixes = tuple(x.upper().strip() for x in args.type_prefix if x.strip())

    downloaded = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for cik in ciks:
            sub_url = SUBMISSIONS_URL.format(cik=cik)
            try:
                submissions = fetch_json(sub_url, args.user_agent)
            except (HTTPError, URLError) as exc:
                print(f"skip cik={cik} submissions_error={exc}", flush=True)
                continue
            time.sleep(args.delay_sec)

            for rec in recent_filings(submissions, allowed_forms, args.limit_filings_per_cik):
                txt_url = filing_txt_url(rec)
                print(
                    f"scanning cik={rec.cik} accession={rec.accession} form={rec.form} date={rec.filing_date}",
                    flush=True,
                )
                try:
                    filing_raw = fetch_text(txt_url, args.user_agent)
                except (HTTPError, URLError) as exc:
                    print(f"skip filing cik={rec.cik} accession={rec.accession} filing_error={exc}")
                    continue
                time.sleep(args.delay_sec)

                docs = parse_document_blocks(filing_raw)
                if not docs:
                    continue

                archive_url = archive_dir_url(rec)
                for doc in docs:
                    if not is_target_exhibit(doc, type_prefixes):
                        continue
                    filename = doc["filename"]
                    exhibit_url = urljoin(archive_url, filename)
                    rel_dir = Path(str(rec.cik)) / rec.accession
                    saved_path = out_dir / rel_dir / safe_name(filename)
                    try:
                        blob = fetch_bytes(exhibit_url, args.user_agent)
                    except (HTTPError, URLError) as exc:
                        print(
                            f"skip exhibit cik={rec.cik} accession={rec.accession} "
                            f"filename={filename} exhibit_error={exc}"
                        , flush=True)
                        continue
                    time.sleep(args.delay_sec)
                    save_bytes(saved_path, blob)
                    meta = ExhibitRecord(
                        cik=rec.cik,
                        accession=rec.accession,
                        form=rec.form,
                        filing_date=rec.filing_date,
                        exhibit_type=doc["type"],
                        exhibit_description=doc.get("description", ""),
                        exhibit_filename=filename,
                        filing_txt_url=txt_url,
                        exhibit_url=exhibit_url,
                        saved_path=str(saved_path),
                    )
                    manifest.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
                    downloaded += 1
                    print(
                        f"downloaded cik={rec.cik} accession={rec.accession} "
                        f"type={doc['type']} -> {saved_path}"
                    , flush=True)
                    if args.download_limit and downloaded >= args.download_limit:
                        print(f"download_limit_reached={downloaded}", flush=True)
                        return


if __name__ == "__main__":
    main()
