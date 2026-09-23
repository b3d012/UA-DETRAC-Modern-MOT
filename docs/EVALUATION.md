# M4 Evaluation Contract

`modern_mot` and `ua_detrac_pr` are separate result families. Ordinary MOTA
is a fixed-operating-point CLEAR metric; PR-MOTA is a UA-DETRAC aggregate over
detector operating points. They must never share a column or aggregate.
Modern quality metrics are normalized to `fraction_0_1`; FP, FN, and IDSW are
integer counts. Native values retain the official toolkit's reported scale.
The pinned TrackEval commit still references NumPy's removed `np.float` and
`np.int` aliases. The adapter restores those aliases in-process immediately
before invoking TrackEval; this is a compatibility shim, not a metric change.

## Canonical inputs

Tracker observations are JSON records with `sequence`, original XML
`frame_number`, positive integer `track_id`, half-open `bbox_xyxy`, and an
optional `[0,1]` score. PR manifests additionally declare each detector
operating point's threshold, recall, precision, and tracker result source.
The document has `schema_version: 1`, `coordinate_convention:
xyxy_half_open`, a `producer` object, `role`, and `observations`.  The CLI
validates duplicate `(sequence, frame, id)` tuples, geometry, score range,
requested split membership, and role before creating evaluator staging.

## Annotation coverage

Only XML-present frames are evaluable. XML-absent image frames are excluded and
reported, never treated as empty negatives. Contiguous annotated spans become
deterministic segment IDs of the form
`<sequence>__annseg_<ordinal>_<start>_<end>`. Associations reset between
segments; result metadata records the policy, segment mapping, and exclusions.
It also records every discarded prediction with sequence, original frame number,
and track ID, so an unknown-coverage frame cannot silently become a false
positive or negative.

## Ignored regions

The verified native workflow calls `dropTracks(stateInfo, curSequence)` in
`trackingEvaluation.m` after tracker execution and before `saveResults`; later
`printFinalEvaluation.m` invokes `DETRAC_MOT_EVAL.exe` on those saved files.
The adapter therefore filters canonical data exactly once when declared
`native_ignore_filter_state=not_applied`; it skips filtering only when declared
`already_applied` with provenance. The rule and asset hashes belong in result
metadata. Modern evaluation applies its independently configured policy during
staging and records that policy plus separate ground-truth and prediction
suppression counts. `none` leaves both sides untouched.

The inspected native `utils/parseGT.m` builds the ignored-region raster map,
removes GT detections with ignored coverage of at least 50%, then retains each
trajectory from its first through last remaining detection. `trackingEvaluation.m`
independently invokes `dropTracks.m` on tracker output before saving it for the
executable. Modern TrackEval uses a deliberately narrower, project-defined
policy: the verified `dropTracks.m` rounding/50% observation rule is applied
symmetrically to individual GT and prediction observations. This avoids an
ignored prediction becoming an FN opportunity while neither deleting ordinary
GT nor claiming equivalence to native trajectory-span cleanup. Native PR
behavior remains isolated and unchanged.

## Native prerequisites

`DETRAC_MOT_EVAL.exe` consumes tracker name, detector name, threshold file,
detection-PR file, sequence list, tracking-result root, and output directory.
It receives no M2 XML ground-truth argument. The adapter fingerprints the
executable, workflow scripts, ignored-region files, and discovered native
assets; it fails if required official assets cannot be established and never
reconstructs native PR ground truth from M2 XML.

## Running and provenance

`evaluate_tracking.py` defaults to the frozen validation role.  It rejects a
sequence outside that role, rejects tuning access to official test, and accepts
official test only with `--mode evaluation --allow-official-test`.  Outputs are
new directories only and contain deterministic `result.json` and `summary.md`.
Every result records the Git state, frozen split/config/lock hashes, command
line, Python/NumPy/SciPy/TrackEval versions, annotation-gap and ignore policy;
native results additionally record toolkit fingerprints and filter state.

The native diagnostic context supplies the exact seven executable arguments
and the official output file location. It must point to the official detector
threshold/PR files and workflow-owned native assets. It is intentionally not
a conversion from M2 XML, and a compatibility diagnostic is not a published
PR-MOTA reproduction unless every detector, PR curve, tracker, preprocessing,
sequence list, and toolkit fingerprint match.

## Native-toolkit acceptance finding

The locally supplied `DETRAC-MOT-toolkit/sequences.txt` contains 60 sequences,
including `MVI_20011`. Historical detector PR files are present for ACF,
compACT, DPM, and R-CNN. No detector-specific `*_thres.txt` file exists under
the supplied `UA_DETRAC_ROOT`; the generic root `thresh.txt` is not a
substitute. Since `printFinalEvaluation.m` requires both
`<detectorName>_thres.txt` and `<detectorName>_detection_PR.txt`, a real native
`DETRAC_MOT_EVAL.exe` compatibility execution is blocked by an external,
missing historical detector artifact. M4 will not fabricate it, rename the
generic file, or reinterpret M2 coverage as the blocker.

Accordingly, the modern TrackEval path is fully validated by dataset-free
known-answer fixtures and real validation GT-as-prediction sanity runs. The
native path has validated adapter/export/parser/invocation contracts, but real
official executable scoring remains unavailable until authentic detector-
specific threshold artifacts are supplied and fingerprinted.

## Local M4 acceptance

Run the focused, dataset-free checks first:

```powershell
python -m pip install -e '.[dev]' -r requirements/evaluation.lock
python scripts/verify_evaluation_prerequisites.py --toolkit-root "$env:UA_DETRAC_ROOT\DETRAC-Toolkits\DETRAC-MOT-toolkit"
pytest -q tests/test_evaluation_core.py tests/test_evaluation_reporting.py tests/test_evaluation_trackeval.py tests/test_evaluation_native.py tests/test_evaluation_native_export.py tests/test_evaluation_native_runner.py
python scripts/evaluate_tracking.py --backend trackeval --synthetic-fixtures --fixture-suite all --output outputs/metrics/m4_synthetic_trackeval
```

`fixture-suite all` emits perfect, empty, false-positive, missed-detection,
ID-switch, detection-gap/fragmentation diagnostic, matched-imperfect-
localization, threshold-crossing localization, legacy-ignore, `none`-ignore,
and interior annotation-gap identity-reset cases. The JSON carries each case's
real TrackEval metrics plus deterministic policy diagnostics; `summary.md`
renders every fixture. `fixture-suite core` is the deliberately smaller first
five-case subset.

Then, only on a local dataset checkout, run the validation-only GT sanity
diagnostic (never a benchmark):

```powershell
python scripts/evaluate_tracking.py --backend trackeval --root "$env:UA_DETRAC_ROOT" --role validation --sequence MVI_20011 --sequence MVI_20032 --sequence MVI_20052 --ground-truth-as-prediction --output outputs/metrics/m4_gt_trackeval_sanity
python scripts/evaluate_tracking.py --backend ua-detrac-pr --root "$env:UA_DETRAC_ROOT" --role validation --sequence MVI_20011 --ground-truth-as-prediction --diagnostic-pr-context configs/evaluation/m4_native_diagnostic_pr.yaml --native-ignore-filter-state not_applied --toolkit-root "$env:UA_DETRAC_ROOT\DETRAC-Toolkits\DETRAC-MOT-toolkit" --output outputs/metrics/m4_native_pr_compatibility
```

Each successful command exits 0. The final two create `result.json` and
`summary.md`; the native diagnostic exits clearly with a prerequisite/config
error until its deliberately placeholder PR context is replaced with matching
official assets.
