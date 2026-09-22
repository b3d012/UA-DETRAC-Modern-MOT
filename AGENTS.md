# UA-DETRAC Modern MOT — Agent Instructions

## Project principle

This repository implements a protocol-faithful research study of detector–tracker compatibility on UA-DETRAC.

The thesis is not a leaderboard-only benchmark. Its core contribution is to explain how the **temporal structure of detector errors** influences multi-object tracking, using both real detector outputs and controlled perturbations.

Work proceeds strictly through the milestones in `docs/ROADMAP.md`.

**Cumulative rule:** `M(n+1) = accepted M(n) + one new capability`.

Do not begin work belonging to a later milestone.

## Required reading before modifying code

Read, in this order:

1. `AGENTS.md`
2. `README.md`
3. `docs/ROADMAP.md`
4. `docs/DATASET_POLICY.md`
5. `docs/REPRODUCIBILITY.md`
6. Existing tests and configs relevant to the requested milestone.

Do not infer project policy from filenames alone.

## Objective

Implement only the milestone explicitly requested by the user.

If M1 is requested, implement M1 only. If M2 is requested, preserve accepted M1 behavior and add only the M2 capability. The same rule applies throughout the roadmap.

## Scope discipline

- Modify only files/directories necessary for the requested milestone.
- Do not implement future-milestone features merely because they are convenient.
- If a future capability requires an interface decision now, create the smallest clean interface necessary without implementing the future capability.
- Preserve accepted behavior from earlier milestones unless the current milestone explicitly requires a compatible extension.

## Immutable research rules

- Preserve the official UA-DETRAC **60 training / 40 testing sequence** partition.
- The official 40-sequence test set is evaluation-only.
- Never use official-test performance for checkpoint selection, threshold selection, early stopping, ablations, detector settings, tracker settings, Re-ID selection, or hyperparameter tuning.
- Validation must be derived only from the official 60 training sequences.
- Splits are always sequence-level, never random frame-level.
- Once a split is frozen by an accepted milestone, do not silently modify it.
- Raw UA-DETRAC files are immutable inputs and must never be modified by project code.
- Raw dataset files must never be committed to Git.
- Do not overwrite frozen experiment outputs or silently replace cached results.
- Do not introduce data leakage between development-train, validation, and official test.
- Do not introduce a second competing canonical annotation representation. External formats must be adapters from the canonical schema.

## Raw dataset location

The recommended local checkout convention is:

```text
data/ua_detrac/
├── DETRAC-Images/
├── DETRAC-Train-Annotations-XML/
├── DETRAC-Test-Annotations-XML/
└── DETRAC-toolkit/   # naming may vary
```

The entire `data/ua_detrac/` directory is ignored by Git.

Code must resolve the dataset through `UA_DETRAC_ROOT` or an explicit config/CLI override. Do not hard-code a machine-specific absolute path.

## Reproducibility requirements

Where applicable, every milestone must provide:

- deterministic behavior given documented seeds;
- configuration-driven execution;
- command-line entry points;
- automated tests;
- concise documentation;
- explicit failure messages rather than silent fallbacks;
- machine-independent paths;
- experiment/configuration metadata;
- stable, machine-readable outputs.

## Existing artifacts

Treat artifacts frozen by accepted milestones as immutable inputs where they already exist.

Examples include:

- dataset manifests;
- frozen split files;
- canonical schema definitions;
- detector checkpoints/configs;
- cached detections;
- tracker configs;
- benchmark outputs;
- final tables and figures.

If an earlier frozen artifact appears incorrect, report it instead of silently rewriting it.

## Testing

Every milestone must include tests appropriate to its functionality.

Do not declare a milestone complete merely because a command runs once. Test normal behavior, important edge cases, deterministic behavior, and relevant research-integrity constraints.

Dataset-backed tests may skip cleanly when the raw dataset is unavailable, but pure unit tests should not require the dataset.

## Stop condition

Stop when the requested milestone's acceptance gate in `docs/ROADMAP.md` passes.

Do not begin the next milestone.

At completion, report:

- files created or changed;
- commands required to reproduce the work;
- tests executed and results;
- acceptance criteria satisfied;
- assumptions made;
- unresolved issues or risks;
- anything intentionally deferred to a later milestone.

## Git convention

Suggested milestone commits:

- `M01: ...`
- `M02: ...`
- `M03: ...`

Suggested accepted milestone tags:

- `m1-foundation`
- `m2-annotations`
- `m3-split-lock`
- `m4-evaluation`
- etc.

Do not rewrite history behind accepted milestone tags.

## Agent handoff pattern

A normal implementation request should be short because the repository already contains the rules:

> Read AGENTS.md, README.md, docs/ROADMAP.md, docs/DATASET_POLICY.md, and docs/REPRODUCIBILITY.md. Implement M1 only. Follow the M1 tasks, non-goals, and acceptance gate exactly. Inspect the repository before editing. Add tests and reproducible CLI commands. Do not begin M2. Stop when the M1 acceptance gate passes and give a completion report.

After implementation, use a separate audit pass before accepting/tagging the milestone.
