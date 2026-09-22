# Beyond mAP — UA-DETRAC Detector–Tracker Compatibility Study

A reproducible research codebase for the thesis project:

> **Beyond mAP: Explaining Detector–Tracker Compatibility in Multi-Object Vehicle Tracking — A Protocol-Faithful Factorial and Detector-Error Study on UA-DETRAC**

The project studies **why** detector outputs affect multi-object trackers differently. The modern detector × tracker benchmark is the empirical foundation; the main explanatory contribution is the analysis of temporal detector-error structure and controlled perturbations.

## Current status

**M1 — Benchmark Foundation** is accepted and tagged `m1-foundation`.
**M2 — Annotation and Visualization Layer** is the current implementation milestone.

Work is governed by:

- [`AGENTS.md`](AGENTS.md) — permanent rules for coding/research agents.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — cumulative M1–M14 implementation plan.
- [`docs/DATASET_POLICY.md`](docs/DATASET_POLICY.md) — split/test-set discipline.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — experiment metadata requirements.

M2 must preserve the accepted M1 foundation and its frozen artifacts.

## Research questions

The study is designed around questions such as:

1. How large are detector, tracker, and detector × tracker interaction effects under a controlled modern UA-DETRAC benchmark?
2. Do temporal detector-error features explain association performance beyond conventional frame-wise metrics such as AP/recall?
3. At matched aggregate detection quality, do burst misses, persistent false positives, localization jitter, and score volatility affect trackers differently?
4. How do these effects vary with UA-DETRAC conditions and computational constraints?

## Benchmark rules

- Preserve the official **60-sequence training / 40-sequence testing** partition.
- Treat the official 40 test sequences as immutable evaluation-only data.
- Derive validation only from the official 60 training sequences and only at sequence level.
- Never choose checkpoints, thresholds, detector settings, tracker settings, Re-ID settings, or ablations using official-test performance.
- Keep raw UA-DETRAC data out of Git.
- Generate all final tables/figures from reproducible result files rather than editing results manually.

## Local dataset convention

The recommended local layout is:

```text
data/
└── ua_detrac/
    ├── DETRAC-Images/
    ├── DETRAC-Train-Annotations-XML/
    ├── DETRAC-Test-Annotations-XML/
    └── DETRAC-toolkit/   # exact downloaded name may vary
```

The entire `data/ua_detrac/` directory is ignored by Git.

Set:

```bash
export UA_DETRAC_ROOT=/absolute/path/to/UA-DETRAC-Modern-MOT/data/ua_detrac
```

Code must also support an explicit CLI/config override and must never silently move raw files.

## M1 — Benchmark Foundation

M1 is intentionally narrow. It verifies the dataset and environment before annotation conversion, data splitting, evaluation, or model integration.

M1 must:

1. resolve `UA_DETRAC_ROOT` or an explicit dataset-root override;
2. identify the existing raw layout without moving files;
3. recognize exactly 60 official training and 40 official testing sequences;
4. match sequence folders to annotation sources;
5. build a deterministic dataset manifest;
6. capture reproducibility/environment metadata;
7. verify raw dataset content is ignored by Git;
8. provide tests and a reproducible verification CLI.

M1 explicitly does **not** create the 48/12 development split, build the canonical annotation schema, integrate TrackEval, or run detectors/trackers. Those belong to later milestones.

### M1 commands

Use Python 3.10–3.12 and install the project with its development dependencies:

```bash
python -m pip install -e '.[dev]'
python scripts/verify_dataset.py --root "$UA_DETRAC_ROOT"
python scripts/capture_environment.py
```

The verification command discovers frame counts, naming conventions, image dimensions, annotation
matches, and top-level sequence metadata from the raw checkout. It does not hard-code observed image
dimensions or parse frame targets. It writes a deterministic, machine-readable manifest to
`data/manifests/ua_detrac_manifest.json`; all stored dataset paths are relative to the dataset root.

To confirm determinism, run verification twice and compare the manifest's SHA-256 digest. The
environment command writes a host-specific JSON snapshot under the ignored `outputs/environment/`
directory. Re-verification treats identical manifest content as a no-op and refuses to overwrite a
different existing manifest; protocol revisions must use a separately reviewed output path.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the complete acceptance gate and agent prompt.

## Milestone workflow

For every milestone:

```text
implement
→ audit
→ fix
→ acceptance gate
→ commit/tag
→ next milestone
```

Suggested accepted tags begin with `m1-foundation`, `m2-annotations`, `m3-split-lock`, `m4-evaluation`, and continue according to the roadmap.

## Reproducibility

The default seed/configuration lives in `configs/reproducibility.yaml`. Every substantive experiment should record the Git state, config hash, seed, software/hardware environment, and command used to produce it.

## Citation

UA-DETRAC data are not redistributed by this repository. Users of the benchmark should cite the official UA-DETRAC publication and follow the dataset's own usage/license terms.
