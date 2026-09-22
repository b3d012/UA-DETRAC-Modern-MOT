# Agent handoff

This repository is governed by `AGENTS.md` and `docs/ROADMAP.md`.

Do not use an old one-off M1 scaffold prompt. Start an agent by asking it to read the repository governance files and implement exactly one milestone.

Example:

> Read AGENTS.md, README.md, docs/ROADMAP.md, docs/DATASET_POLICY.md, and docs/REPRODUCIBILITY.md. Implement M1 only. Follow the M1 tasks, non-goals, and acceptance gate exactly. Inspect the repository before editing. Add tests and reproducible CLI commands. Do not begin M2. Stop when the M1 acceptance gate passes and give a completion report.

After implementation, perform a separate audit pass before tagging the milestone.
