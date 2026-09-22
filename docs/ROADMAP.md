# UA-DETRAC Thesis Implementation Roadmap

**Project:** Beyond mAP: Explaining Detector–Tracker Compatibility in Multi-Object Vehicle Tracking  
**Repository:** `UA-DETRAC-Modern-MOT`  
**Core rule:** every milestone is cumulative. Do not begin milestone `M(n+1)` until the acceptance gate for `M(n)` passes and the milestone is committed/tagged.

The benchmark matrix is experimental infrastructure. The thesis-specific explanatory contribution is primarily **M11–M13**, where temporal detector errors are measured and manipulated.

## Milestone dependency map

```text
M1  Dataset foundation
 ↓
M2  Annotation + visualization layer
 ↓
M3  Train/validation split + test-lock policy
 ↓
M4  Evaluation harness
 ↓
M5  Detector API + one end-to-end baseline
 ↓
M6  Four-detector suite + frozen detection caches
 ↓
M7  Tracker API + SORT/ByteTrack/OC-SORT
 ↓
M8  StrongSORT/BoT-SORT + Re-ID fairness policy
 ↓
M9  Validation factorial + equal-budget tuning + config freeze
 ↓
M10 Official 4×5 factorial benchmark
 ↓
M11 Detector Error Signature (DES)
 ↓
M12 Controlled perturbation engine
 ↓
M13 Causal sensitivity + statistical interaction analysis
 ↓
M14 Condition analysis + efficiency + publication/result freeze
```

## Repository contract

Use the existing structure rather than parallel ad-hoc folders:

```text
UA-DETRAC-Modern-MOT/
├── AGENTS.md
├── configs/
│   ├── dataset/
│   └── experiments/
├── data/
│   ├── ua_detrac/          # local raw data; NEVER committed
│   ├── manifests/
│   └── splits/
├── docs/
├── experiments/
├── outputs/
│   ├── detections/
│   ├── tracks/
│   ├── metrics/
│   └── figures/
├── scripts/
├── src/
│   ├── datasets/
│   ├── detectors/
│   ├── trackers/
│   ├── evaluation/
│   └── analysis/
└── tests/
```

`UA_DETRAC_ROOT` may point to `data/ua_detrac` in a local checkout. Raw data remain ignored by Git. Every generated result must be reproducible from committed code/configuration and a recorded seed.

---

# M1 — Benchmark Foundation

**Estimated focused time:** 1–2 days.  
**Purpose:** prove that the dataset layout and execution environment are understood before annotation conversion or model code is introduced.

### Tasks

1. Resolve the dataset root from `UA_DETRAC_ROOT` or an explicit CLI/config path.
2. Detect the local UA-DETRAC layout without moving files.
3. Enumerate official train/test sequence folders and match them to annotation files.
4. Record per-sequence frame count, frame naming pattern, image dimensions, annotation filename, and available top-level metadata.
5. Produce a deterministic machine-readable dataset manifest under `data/manifests/`.
6. Capture environment metadata: OS, Python, PyTorch, CUDA, GPU, CPU, core package versions, Git commit, timestamp.
7. Verify that `data/ua_detrac/` and all raw dataset content are ignored by Git.
8. Add smoke/unit tests that do not load the entire image corpus into memory.

### Suggested files

```text
src/datasets/paths.py
src/datasets/manifest.py
scripts/verify_dataset.py
scripts/capture_environment.py
configs/dataset/ua_detrac.yaml
data/manifests/ua_detrac_manifest.json
tests/test_dataset_layout.py
```

### Required CLI

```bash
python scripts/verify_dataset.py --root "$UA_DETRAC_ROOT"
```

Expected concise summary:

```text
Train sequences: 60
Test sequences: 40
Sequences with missing frames: 0
Sequences without annotations: 0
Manifest: data/manifests/ua_detrac_manifest.json
STATUS: PASS
```

### Acceptance gate

M1 passes only when:

- exactly 60 official training sequences and 40 official test sequences are recognized;
- every expected sequence has images and its matching annotation source;
- the manifest is deterministic across two consecutive runs;
- tests pass;
- Git reports no raw dataset file as tracked/staged.

### Non-goals

Do not convert annotations, create a validation split, integrate TrackEval, train models, or implement trackers.

### Agent prompt

> Implement M1 Benchmark Foundation only. Verify the existing UA-DETRAC raw layout under `data/ua_detrac`, create a deterministic dataset manifest and environment capture, add tests, and stop once the M1 acceptance gate passes. Do not move raw data or begin M2.

---

# M2 — Annotation and Visualization Layer

**Estimated focused time:** 2–3 days.  
**Purpose:** establish one trustworthy internal ground-truth representation used by all later components.

### Tasks

1. Read/document the official XML structure and bounding-box semantics.
2. Parse frame targets, track IDs, vehicle type, occlusion/truncation information, ignored regions, and available sequence attributes.
3. Create one canonical internal schema/dataclass.
4. Use an explicit coordinate convention, preferably `xyxy`, with tested source conversion helpers.
5. Validate positive box area, image ranges, frame continuity, class vocabulary, track-ID uniqueness per frame, and XML/image alignment.
6. Build GT visualization with boxes, IDs, classes, ignored regions, and relevant metadata.
7. Produce a visual audit set covering multiple sequences/conditions.

### Suggested files

```text
src/datasets/schema.py
src/datasets/ua_detrac.py
src/datasets/validation.py
scripts/validate_annotations.py
scripts/visualize_ground_truth.py
tests/test_annotations.py
tests/test_box_conversions.py
```

### Acceptance gate

- parser/unit tests pass;
- malformed/impossible boxes are either absent or explicitly explained;
- at least 30–50 frames are visually audited;
- track IDs visibly persist across adjacent frames;
- ignored regions are handled consistently;
- exactly one canonical annotation schema exists.

### Non-goals

Do not create the development split or detector-specific training labels.

---

# M3 — Sequence Split and Leakage Lock

**Estimated focused time:** 0.5–1 day.  
**Purpose:** make test contamination technically difficult.

### Tasks

1. Split only the official 60 training sequences into 48 development-train and 12 validation sequences.
2. Use a fixed seed and record the selection algorithm.
3. Balance available sequence-level conditions where feasible; document distributions.
4. Write frozen split manifests under `data/splits/`.
5. Expose roles: `development_train`, `validation`, `official_test`.
6. Reject tuning-mode configurations that reference official-test sequences.
7. Add overlap/leakage tests.

### Acceptance gate

- counts are exactly 48/12/40;
- all intersections are empty;
- split files are deterministic and committed;
- leakage guards are tested;
- no model performance influenced split selection.

### Non-goals

Do not optimize the split using model scores.

---

# M4 — Evaluation Harness

**Estimated focused time:** 2–4 days.  
**Purpose:** prove metric correctness before trusting model scores.

### Tasks

1. Integrate TrackEval or an equivalently validated evaluator for HOTA, DetA, AssA, IDF1 and CLEAR MOT metrics.
2. Integrate/wrap the official UA-DETRAC PR-based evaluation for historical compatibility.
3. Convert canonical data to evaluator formats.
4. Create fixtures: perfect prediction, empty prediction, one miss, one FP, one ID switch, controlled localization error.
5. Verify expected metric direction/value.
6. Emit machine-readable and concise human-readable evaluation summaries.
7. Document ordinary MOTA versus PR-MOTA so they are never mixed in a table.

### Acceptance gate

- perfect predictions score as expected;
- synthetic error fixtures change metrics as expected;
- format conversions are tested;
- GT-as-prediction succeeds on several real sequences;
- both modern and UA-DETRAC evaluation paths are validated.

---

# M5 — Detector API and One End-to-End Baseline

**Estimated focused time:** 2–3 days plus compute.  
**Purpose:** freeze a detector/cache contract before integrating multiple frameworks.

### Tasks

1. Define common per-frame detections: frame, box, confidence, class.
2. Define cache metadata: detector/version, checkpoint/config hashes, source commit, image size, threshold/NMS policy, split, timestamp, hardware.
3. Integrate one baseline detector behind the adapter.
4. Keep model-specific label conversion inside the detector adapter.
5. Train only on development-train and choose checkpoint/threshold on validation.
6. Cache pre-tracker detections.
7. Add prediction-vs-GT visualization.

### Acceptance gate

One detector can train/load, infer, cache, visualize and evaluate entirely through the common contract; trackers can later read caches without importing its framework.

---

# M6 — Core Four-Detector Suite

**Estimated focused time:** roughly 1–2 calendar weeks depending on GPU.  
**Core set:** Faster R-CNN R50-FPN, YOLOv8-S, RT-DETR-R18, YOLOv12-S.  
**Optional later anchor:** YOLOv10-S.

### Tasks

1. Implement each detector behind M5.
2. Record pretrained-weight provenance/version/license.
3. Create comparable training configs and document unavoidable framework differences.
4. Use the same frozen 48/12 split.
5. Freeze resolution, augmentation, seed and training-selection conventions.
6. Record validation curves and selected checkpoint rationale.
7. Cache predictions after configuration freeze.
8. Produce AP50, mAP50:95, precision, recall, per-condition diagnostics, latency, VRAM and parameter reports.

### Acceptance gate

- all four emit the same cache schema;
- each has a frozen checkpoint/config;
- validation reports exist;
- caches are hashable/immutable;
- no official-test result selected a checkpoint or threshold.

---

# M7 — Tracker API + SORT/ByteTrack/OC-SORT

**Estimated focused time:** 2–4 days.  
**Purpose:** integrate tracking independently from detector code.

### Tasks

1. Tracker input is cached detections only.
2. Define common MOT output schema.
3. Integrate SORT, ByteTrack and OC-SORT.
4. Validate each first using ground-truth detections.
5. Confirm deterministic behavior for fixed configs.
6. Expose meaningful tracker parameters via config.
7. Add track/ID visualization.

### Acceptance gate

All three trackers work with GT and cached detector streams, evaluate without manual conversion, and pass empty-frame/low-score/lost-track smoke tests.

---

# M8 — StrongSORT and BoT-SORT + Re-ID Fairness

**Estimated focused time:** 3–5 days.  
**Purpose:** finish the core tracker set while controlling appearance-model confounds.

### Tasks

1. Integrate StrongSORT and BoT-SORT.
2. Record Re-ID checkpoint/domain/provenance.
3. Determine whether a common vehicle Re-ID backbone can be used fairly.
4. If feasible, support `canonical` and `standardized_reid` modes.
5. Never train Re-ID using official-test frames/identities.
6. Cache embeddings if feature extraction is expensive.

### Acceptance gate

All five trackers share one interface; every Re-ID dependency is traceable; the fairness policy clearly states what differs between trackers besides association logic.

---

# M9 — Validation Factorial + Equal-Budget Tuning

**Estimated focused time:** 3–5 days plus compute.  
**Purpose:** freeze fair pair configurations before official-test evaluation.

### Tasks

1. Run the 4×5 combinations on the 12 validation sequences.
2. Predeclare a small tuning space for each tracker family.
3. Use equal search budgets where feasible.
4. Select primarily by HOTA; use AssA/IDF1 as diagnostics.
5. Run detector-confidence threshold sensitivity curves.
6. Log every trial.
7. Freeze the final 20 pair configs and hashes.
8. Create `FROZEN_TEST_CONFIGS.md`.

### Acceptance gate

Exactly 20 final pair configurations are frozen, trial logs are complete, no official-test output informed selection, and reruns reproduce validation results within tolerance.

---

# M10 — Official 4×5 Factorial Benchmark

**Estimated focused time:** 2–4 days once detections are cached.  
**Purpose:** produce the controlled empirical matrix needed for interaction/error analysis.

### Tasks

1. Run the 20 frozen pairs on all 40 official test sequences.
2. Compute per-sequence and aggregate HOTA, DetA, AssA, IDF1, MOTA, FP, FN, IDSW and applicable MT/ML.
3. Run PR-MOTA separately.
4. Measure detector latency, tracker overhead, end-to-end latency/FPS and GPU memory on one declared stack.
5. Save raw per-sequence metrics.
6. Generate all 4×5 tables/heatmaps automatically.
7. Never alter configs after seeing test results.

### Acceptance gate

All runs are complete, aggregate tables rebuild from raw results, config hashes match M9, and the repository records that no post-test tuning occurred.

---

# M11 — Detector Error Signature (DES)

**Estimated focused time:** 4–7 days.  
**Purpose:** characterize detector errors temporally instead of reducing them to frame-wise AP/recall.

### Core DES families

- miss-run length / burst statistics;
- false-positive persistence;
- localization jitter;
- confidence volatility and threshold crossings;
- calibration summaries;
- duplicate tendency;
- condition-specific recall/error rates.

### Tasks

1. Match real detector outputs to GT using a documented assignment rule.
2. Build per-object temporal miss sequences.
3. Compute miss-run distributions and summary statistics.
4. Track unmatched predictions temporally to estimate FP persistence.
5. Measure frame-to-frame localization residual/jitter for matched tracks.
6. Measure confidence variance, threshold crossings and calibration.
7. Quantify duplicates.
8. Emit per-sequence/per-detector DES tables with tests.

### Acceptance gate

Definitions are documented; features are deterministic; hand-constructed sequences produce expected DES values; every core detector has per-sequence signatures.

---

# M12 — Controlled Perturbation Engine

**Estimated focused time:** 3–5 days.  
**Purpose:** create matched-quality streams that differ in one controlled temporal error mechanism.

### Required perturbations

- **P1 Independent false negatives:** randomly drop detections to target recall.
- **P2 Burst false negatives:** same number of drops, clustered into 2/3/5/10-frame runs.
- **P3 Localization noise:** controlled center/scale perturbation with target error severity.
- **P4 Persistent false positives:** compare persistent synthetic trajectories with the same total number of isolated FPs.
- **P5 Confidence corruption:** keep geometry fixed while perturbing/miscalibrating scores.
- **P6 Duplicates:** controlled duplicate boxes around a subset of true detections.

### Design rules

- deterministic from `{sequence, seed, severity}`;
- matched experiments hold aggregate quantity/quality constant as closely as possible;
- never modify GT files;
- emit the normal detector-cache schema;
- severity grids are predeclared.

### Acceptance gate

Matched-recall miss streams agree within tight tolerance; isolated/persistent FP streams match FP count; localization severities meet target budgets; confidence-only perturbations leave geometry unchanged; outputs regenerate deterministically.

---

# M13 — Causal Sensitivity and Statistical Interaction Analysis

**Estimated focused time:** ~1 week.  
**Purpose:** convert the benchmark into explanatory evidence.

### A. Real detector × tracker interaction

Use per-sequence M10 results. Compare:

```text
Additive:     metric ~ detector + tracker + sequence controls
Interaction:  metric ~ detector * tracker + sequence controls
```

Primary outcomes: HOTA, AssA, IDF1. Treat sequence as the unit of analysis.

### B. Does DES explain more than AP/recall?

Compare nested models:

```text
Model 1: association_metric ~ AP/recall
Model 2: association_metric ~ AP/recall + DES features
```

Use sequence-level cross-validation/bootstrap.

### C. Matched-recall misses

Compare independent versus burst misses at equal recall. Measure AssA, IDF1, IDSW and HOTA.

### D. FP persistence

Compare isolated versus persistent FPs with matched total FP count.

### E. Localization jitter

Sweep localization noise while detection presence remains fixed.

### F. Score volatility

Keep boxes unchanged and corrupt confidence, particularly probing score-sensitive trackers such as ByteTrack.

### Statistical rules

- sequence-level bootstrap confidence intervals;
- effect sizes and intervals, not only p-values;
- family-wise correction such as Holm where appropriate;
- bootstrap/permutation methods when assumptions are poor;
- predeclare primary outcomes;
- save long-format analysis tables and scripts for every figure.

### Acceptance gate

RQ1–RQ3 each have a direct reproducible result table/figure, uncertainty estimates, and interpretations that clearly distinguish observational evidence from controlled perturbation evidence.

---

# M14 — Condition Analysis, Efficiency, Robustness, and Publication Freeze

**Estimated focused time:** 3–5 days plus writing.  
**Purpose:** answer condition/efficiency questions and freeze publication-ready evidence.

### Tasks

1. Stratify results by validated UA-DETRAC metadata such as illumination/weather, occlusion and scale/difficulty.
2. Repeat key interaction/error analyses where sample size supports it.
3. Produce accuracy-efficiency plots from same-machine measurements.
4. Run predefined robustness checks:
   - alternative DES IoU matching threshold;
   - alternative detector confidence points;
   - canonical vs standardized Re-ID if available;
   - optional YOLOv10-S replication anchor if explicitly approved.
5. Perform a clean reproduction for all headline outputs.
6. Freeze final tables/figures with experiment IDs.
7. Draft Methods/Results from generated outputs.

### Acceptance gate

RQ4 is answered; central conclusions survive reasonable robustness checks or limitations are explicitly documented; every headline table/figure is script-generated; final result artifacts are frozen for manuscript/thesis writing.

---

# First two weeks — practical sequential target

| Day | Target | End-of-day result |
|---|---|---|
| 1 | M1 dataset-root/layout resolver | Repo locates all four raw assets |
| 2 | M1 manifest/tests/environment | M1 accepted/tagged |
| 3 | M2 XML parser/schema | Canonical annotations load |
| 4 | M2 validation + GT visualization | M2 accepted after visual audit |
| 5 | M3 deterministic 48/12 split + guards | Split frozen |
| 6 | M4 modern evaluator | HOTA/IDF1/MOTA fixtures work |
| 7 | M4 UA-DETRAC wrapper | PR-based path works |
| 8 | M4 synthetic tests/docs | M4 accepted |
| 9 | M5 detector interface/cache | Common detection format frozen |
| 10 | M5 first detector | One sequence end-to-end |
| 11 | M5 validation run/visualization | M5 accepted |
| 12 | M6 detector adapter #2 | Integration verified |
| 13 | M6 detector adapter #3 | Training/cache path verified |
| 14 | M6 detector adapter #4 / training queue | Core suite implementation underway |

Do not weaken the protocol to force M6 training into an arbitrary deadline.

## Safe parallelism later

After interfaces/configs are frozen, expensive compute can be distributed: detector training in M6, frozen official-test pairs in M10, and perturbation sweeps in M12/M13. Parallel jobs must still write the same normalized schemas and never change one another's configuration.

## Thesis-ready success chain

```text
correct dataset
→ trusted annotations
→ leakage-safe split
→ trusted metrics
→ comparable detector outputs
→ comparable tracker outputs
→ frozen factorial benchmark
→ quantified temporal detector errors
→ controlled error interventions
→ sequence-level statistical explanation
→ reproducible figures/tables
```

The strongest final result is not necessarily a winning detector–tracker pair. The strongest result is a defensible mechanism-level conclusion, for example: **at matched detector recall, temporally clustered misses may cause greater association degradation than independent misses, with tracker families showing different sensitivities to that temporal structure.**
