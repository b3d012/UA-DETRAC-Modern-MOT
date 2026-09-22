# Scripts

M1 provides two thin command-line entry points:

```bash
python scripts/verify_dataset.py --root "$UA_DETRAC_ROOT"
python scripts/capture_environment.py
```

`verify_dataset.py` validates the raw layout and writes the deterministic dataset manifest.
`capture_environment.py` writes a host-specific, Git-ignored environment snapshot. Dataset
download, annotation visualization, split generation, evaluation, and model commands belong to
later milestones.

Keep reusable logic in `src/`; scripts should primarily parse arguments and call library functions.
