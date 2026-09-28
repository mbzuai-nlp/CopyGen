# Data Generation Scripts

The released benchmark is already available under `data/`. These scripts document how the four benchmark domains were constructed and can be used to rebuild intermediate files under `outputs/generated/`.

- Contract: `generate_legal_clause_tasks.py`, `generate_legal_clause_text_tasks.py`
- Literature: `generate_pmc_structured_span_tasks.py`
- Code: `generate_python_fragile_span_tasks.py`
- Random Sequence: `ssh/generate_secret_copy_tasks.py`

Final released domain names are:

```text
data/contract/
data/literature/
data/code/
data/random_sequence/
```

Keep generated intermediates outside `data/` unless they are the final released train/test files.
