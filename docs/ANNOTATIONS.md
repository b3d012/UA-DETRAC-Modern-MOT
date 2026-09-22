# UA-DETRAC Annotation Contract (M2)

M2 defines the repository's only canonical ground-truth representation. Future
detector-label and evaluator formats must be adapters from this schema rather
than alternative internal representations.

## Official XML mapping

Each annotation file has a `sequence` root with its source `name`, optional
`sequence_attribute`, zero or more `ignored_region/box` elements, and `frame`
elements. A frame contains `num`, source `density`, and zero or more
`target_list/target` elements. Targets carry an `id`, a `box`, an `attribute`
element (including `vehicle_type` and available source values such as
`truncation_ratio`, orientation, speed, and trajectory length), plus optional
`occlusion/region_overlap` elements.

All represented XML attributes, including `sequence`, `ignored_region`, and
`occlusion` container attributes, are retained as sorted immutable `(name, value)` pairs.
This intentionally retains original spellings, including the official
`sence_weather` attribute. Typed fields are derived for frames, IDs, boxes,
and known occlusion IDs/statuses; the raw value remains available for
provenance.

## Coordinates

UA-DETRAC source boxes are retained as floating-point `left`, `top`, `width`,
and `height` (`xywh`). Canonical geometry derives a floating-point, half-open
box `(x1, y1, x2, y2) = (left, top, left + width, top + height)`. The parser
does not round, clip, shift for pixel indexing, or otherwise normalize source
coordinates. Validation reports impossible areas and out-of-image coordinates
instead of correcting them.

Official train annotations contain legitimate truncated/border boxes whose
source-derived right or bottom extent reaches one pixel beyond a 960×540 image
(for example, `x2=961` or `y2=541`). M2 accepts an extent no greater than one
pixel beyond the right/bottom image boundary while retaining the exact
unclamped canonical box. Larger positive overflow, negative coordinates, and
non-positive or non-finite dimensions remain validation errors. Consumers that
need strict image-space geometry must call `clip_box_to_image(box, width=...,
height=...)`; this helper returns clipped coordinates and never changes the
canonical annotation.

## Validation and visualization

Routine commands default to the official training partition. A sequence must
be named explicitly unless the user deliberately requests `--all`. Any access
to official-test annotations requires `--allow-official-test`.

```powershell
python scripts/validate_annotations.py --root "$env:UA_DETRAC_ROOT" --partition train --sequence MVI_20011
python scripts/visualize_ground_truth.py --root "$env:UA_DETRAC_ROOT" --audit --output outputs/figures/m2_ground_truth_audit
```

`--audit` reads the frozen M1 manifest, selects official-training sequences
deterministically across available `(sence_weather, camera_state)` conditions,
and renders 40 consecutive-frame samples whenever enough eligible data exists.
It adds train sequences when fewer than five condition groups exist; it warns
explicitly if fewer than 30 frames exist. Rendered output displays boxes,
track IDs, classes, ignored regions, sequence/frame metadata, truncation, and
occlusion flags. Existing nonempty output directories are rejected unless
`--force` is supplied.

Both commands resolve `--root` first and otherwise use `UA_DETRAC_ROOT`.

## Annotation coverage semantics

An image frame with an XML `<frame>` element is an **annotated frame**. An
image-present frame without an XML `<frame>` element is **unannotated**: its
annotation coverage is unknown and it is not an implicit zero-object or
negative frame. The immutable `annotation_coverage` metadata preserves the
exact image/XML frame-number sets and their differences.

Normal validation reports image frames absent from XML as the non-fatal
`image_frame_absent_from_xml` coverage warning. XML frames without a matching
image remain the fatal `xml_frame_without_image` integrity error. The
validator's JSON report retains total, error, and warning counts; it exits
successfully when warnings are the only findings.

Future detector-training adapters must not automatically use XML-absent image
frames as negative examples. Their training policy is deliberately deferred
until the detector dataset adapter is designed.

### Frame-alignment diagnostics

Use `--diagnose-frame-alignment` only with explicit sequence names to inspect
raw XML/image frame-set differences. It reports counts and bounds, missing-set
counts, first/last 20 image frame numbers absent from XML, and contiguous
leading/trailing/interior missing blocks. It is diagnostic only: it neither
suppresses `xml_image_frame_mismatch` during normal validation nor interprets
missing XML frames as empty ground-truth frames.

```powershell
python scripts/validate_annotations.py --partition train --diagnose-frame-alignment --sequence MVI_39761
```
