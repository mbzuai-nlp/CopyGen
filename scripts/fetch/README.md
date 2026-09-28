# Fetch Scripts

These utilities fetch or clean raw source material used by benchmark construction.

- `fetch_edgar_exhibits.py`: download SEC EDGAR exhibits used for contract sources.
- `clean_edgar_contracts.py`: filter and normalize contract text.
- `fetch_pmc_packages.py`: download PMC Open Access packages.
- `extract_pmc_packages.py`: unpack PMC packages into text and NXML sources.

The Code and Random Sequence domains do not require these fetchers: Code is built from local source snapshots, and Random Sequence is synthetically generated.
