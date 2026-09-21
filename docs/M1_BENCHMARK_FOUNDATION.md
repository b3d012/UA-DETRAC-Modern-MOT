# M1 — Benchmark Foundation

## Goal

Establish a clean, testable, reproducible UA-DETRAC research environment before attempting to improve detector or tracker performance.

## Deliverables

### Dataset acquisition and verification

- Document official/author-maintained download sources.
- Download/extract without committing data.
- Verify 60 official train sequences and 40 official test sequences.
- Record file sizes/checksums for downloaded archives when feasible.
- Detect incomplete/corrupt extraction early.

### Annotation and metadata parsing

Create a normalized representation for:

- sequence name and sequence-level attributes;
- frame number;
- track/object ID;
- bounding box;
- vehicle type;
- truncation;
- occlusion information when available in the source annotation;
- orientation and any other official fields present;
- ignored regions.

The parser must preserve raw values and make any derived/convenience representation explicit.

### Validation

Validate at minimum:

- expected sequence membership/counts;
- frame filename continuity and counts;
- image readability and dimensions;
- XML/frame alignment;
- legal/known classes;
- unique object IDs per frame;
- bbox width/height positivity;
- bbox coordinate consistency with image boundaries, while correctly handling officially valid truncation semantics;
- ignored-region format;
- missing/duplicate frames and annotations.

Emit both human-readable summaries and machine-readable JSON/CSV manifests.

### Visualization

Provide a CLI to render a frame with:

- GT boxes;
- track IDs;
- vehicle class;
- relevant attributes;
- ignored regions;
- sequence/frame metadata.

It should support saving an image to `outputs/figures/` and optionally opening an interactive window.

### Train/validation/test protocol

- Keep official 40 test sequences untouched.
- Create deterministic train/validation manifests from the 60 official-train sequences only.
- Store the exact split manifest and seed in version control once finalized.
- Include an assertion/test that train, validation, and official test sequence sets are disjoint.

### Environment and experiment identity

Implement one command that records system/library versions plus Git/config metadata to a JSON file under the experiment directory.

### Tests

Unit tests should cover parser edge cases, bbox conversion, deterministic splitting, split leakage checks, manifest generation, and environment capture behavior that can be tested without a GPU/dataset.

Dataset-backed smoke tests may skip cleanly when `UA_DETRAC_ROOT` is absent.

## Definition of done

M1 is done when a fresh clone plus a valid `UA_DETRAC_ROOT` can run a documented setup/validation sequence and produce deterministic manifests, visual sanity checks, and a reproducibility record without training a detector or tracker.
