# Dataset Policy

## Non-negotiable split discipline

UA-DETRAC's official training and testing partitions are benchmark assets, not suggestions.

- Official train: 60 sequences.
- Official test: 40 sequences.
- The official test set remains evaluation-only.
- Validation is derived exclusively from the 60 official training sequences.
- Splitting is performed at **sequence level**, never frame level, to avoid temporal leakage.
- No checkpoint, threshold, early-stopping decision, detector confidence/NMS setting, tracker association setting, Re-ID choice, ablation, or model-selection decision may be selected using official-test performance.

## Raw data

Raw frames, annotations, archives, and toolkit files are never committed to Git.

For this repository, the recommended local location is `data/ua_detrac/`. That directory is ignored in `.gitignore`.

Project code must reference the dataset through `UA_DETRAC_ROOT` or an explicit config/CLI override. Raw files are immutable inputs: scripts must not silently rename, reorganize, overwrite, or normalize them in place.

## Milestone ownership

- **M1** verifies the raw layout and sequence counts.
- **M2** creates the canonical annotation representation.
- **M3** creates and freezes the 48/12 development-train/validation split from the official 60 training sequences.
- **M9** freezes detector–tracker tuning configurations.
- **M10** performs the primary official-test factorial evaluation.

Earlier milestones must not pre-empt the responsibilities of later milestones.

## Test-set access

Exploratory/tuning workflows should default to development-train/validation data. Once M3 exists, code paths marked as tuning must reject official-test sequences.

Official-test evaluation must be deliberate and reproducible. From M10 onward, every official-test run should be associated with an experiment/config identity so the final result set is auditable.

## Derived artifacts

Generated manifests, split files, cached detections, tracker outputs, metrics, and figures are not raw dataset files. They are governed by the milestone that creates them.

Once an artifact is explicitly frozen by an accepted milestone, later code must treat it as immutable unless a documented protocol revision is approved.
