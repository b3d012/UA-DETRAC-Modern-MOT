# M3 Sequence Split and Test Lock

`data/splits/ua_detrac_m3_split.json` is the frozen M3 partition. It derives
the 48/12 development-train/validation split only from official-training
sequence metadata and XML annotation availability. The official 40 test
sequences are copied unchanged and evaluation-only.

The optimizer accepts only sequence name, `sence_weather`, `camera_state`,
XML-present annotated-frame count, and raw image-frame count. XML-absent
image frames are unknown coverage, never negative examples. It never receives
object counts, classes, boxes, occlusion, truncation, detector/tracker output,
or performance data.

The audit records target and achieved proportions with percentage-point
deviations for categorical conditions and both raw/annotated exposure. Any
non-empty joint condition with zero validation sequences is explicit.

Tuning must use only `development_train` and/or `validation`. The split-access
validator rejects all official-test role or sequence requests in tuning mode.
