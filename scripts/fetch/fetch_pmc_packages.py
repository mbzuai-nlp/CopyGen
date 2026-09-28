#!/usr/bin/env python3
"""
Download a sampled subset of PMC Open Access article packages listed in oa_file_list.csv.

Example:
  python scripts/fetch/fetch_pmc_packages.py \
    --oa-file-list data/pmc/oa_file_list.csv \
    --out-dir data/pmc/packages \
    --limit 50 \
    --citation-regex "chem|catal|experiment|materials" \
    --user-agent "CopyGen research you@example.com"
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen


PMC_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"


def fetch_bytes(url: str, user_agent: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def iter_filter_values(values: list[str], path: str | None) -> list[str]:
    out = list(values)
    if path:
        out.extend(
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected PMC OA article packages.")
    parser.add_argument("--oa-file-list", default="data/pmc/oa_file_list.csv")
    parser.add_argument("--out-dir", default="data/pmc/packages")
    parser.add_argument("--pmcid", action="append", default=[])
    parser.add_argument("--pmcid-file", default="")
    parser.add_argument("--license-allow", action="append", default=[])
    parser.add_argument("--citation-regex", action="append", default=[])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--delay-sec", type=float, default=0.1)
    parser.add_argument("--user-agent", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_pmcids = {x.strip().upper() for x in iter_filter_values(args.pmcid, args.pmcid_file or None)}
    allowed_licenses = {x.strip().upper() for x in args.license_allow if x.strip()}
    citation_patterns = [re.compile(pat, flags=re.I) for pat in args.citation_regex if pat.strip()]

    downloaded = 0
    with Path(args.oa_file_list).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmcid = row["Accession ID"].strip().upper()
            citation = row["Article Citation"].strip()
            license_name = row["License"].strip().upper()
            rel_path = row["File"].strip()

            if wanted_pmcids and pmcid not in wanted_pmcids:
                continue
            if allowed_licenses and license_name not in allowed_licenses:
                continue
            if citation_patterns and not any(p.search(citation) for p in citation_patterns):
                continue

            url = PMC_BASE + rel_path
            filename = Path(rel_path).name
            save_path = out_dir / filename
            if save_path.exists():
                print(f"exists {save_path}")
                continue

            blob = fetch_bytes(url, args.user_agent)
            save_path.write_bytes(blob)
            downloaded += 1
            print(f"downloaded {pmcid} -> {save_path}")
            time.sleep(args.delay_sec)

            if args.limit and downloaded >= args.limit:
                break


if __name__ == "__main__":
    main()
