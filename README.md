# UA-DETRAC Modern MOT

A reproducible research codebase for benchmarking modern object detectors and multi-object trackers on **UA-DETRAC** while preserving the official benchmark protocol.

## Milestone status

This repository is currently at **M1 — Benchmark Foundation**. M1 is infrastructure only: dataset acquisition/verification, parsing, visualization, validation, split discipline, environment capture, and reproducibility. **No model comparison claims or test-set tuning belong in M1.**

## Benchmark rules

- Preserve the official **60-sequence training / 40-sequence testing** partition.
- The official test set is treated as **immutable evaluation-only data**.
- Any validation split must be derived **only from the 60 official training sequences**, at sequence level.
- Never choose hyperparameters, thresholds, detector settings, tracker settings, or model checkpoints using the official test set.
- Raw UA-DETRAC files live outside Git and are addressed through configuration/environment variables.

## Repository layout

```text
UA-DETRAC-Modern-MOT/
├── .github/
├── configs/
│   ├── dataset/
│   └── experiments/
├── data/
│   ├── manifests/
│   └── splits/
├── docs/
├── experiments/
├── src/
│   ├── datasets/
│   ├── detectors/
│   ├── trackers/
│   ├── evaluation/
│   └── analysis/
├── outputs/
│   ├── detections/
│   ├── tracks/
│   ├── metrics/
│   └── figures/
├── scripts/
├── tests/
├── pyproject.toml
└── README.md
```

## Local dataset convention

Set an environment variable pointing at the extracted dataset root:

```bash
export UA_DETRAC_ROOT=/absolute/path/to/UA-DETRAC
```

Do **not** place raw images/annotations in Git. The eventual M1 setup script should support either the official layout directly or a small number of clearly documented legacy layouts without silently moving files.

## M1 acceptance criteria

M1 is complete only when the project can:

1. verify the downloaded UA-DETRAC assets and expected official train/test sequence counts;
2. parse official annotations and sequence-level attributes into a normalized internal representation;
3. validate frame continuity, image dimensions, annotation ranges, class names, track IDs, and bounding-box semantics;
4. visualize frames with GT boxes, track IDs, ignored regions, and relevant attributes;
5. generate a deterministic sequence-level train/validation split from the official 60 training sequences while leaving all 40 test sequences untouched;
6. write a dataset manifest with sequence names, frame counts, resolution, classes, annotation statistics, and checksums where practical;
7. capture CPU, GPU, CUDA, Python, PyTorch, package versions, Git commit, config hash, seed, and experiment ID;
8. run automated tests and validation commands without relying on the test set for tuning.

See [`docs/M1_BENCHMARK_FOUNDATION.md`](docs/M1_BENCHMARK_FOUNDATION.md) and [`docs/DATASET_POLICY.md`](docs/DATASET_POLICY.md).

## Reproducibility

The default seed/configuration lives in `configs/reproducibility.yaml`. A machine-specific dependency lock should be generated during M1 after the Python/PyTorch/CUDA environment is selected, rather than committing an arbitrary CUDA-specific PyTorch wheel choice before the target machine is known.

## Citation

If you use UA-DETRAC, cite the official UA-DETRAC publication and follow the dataset's own usage/license terms. This repository does not redistribute UA-DETRAC data.
