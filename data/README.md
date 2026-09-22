# Data directory

Raw UA-DETRAC files do not belong in Git.

This directory is reserved for small, reviewable research metadata such as:

- exact train/validation/test sequence manifests;
- checksums and download manifests;
- normalized dataset summary tables;
- schema documentation.

M1 generates `manifests/ua_detrac_manifest.json`. It contains dataset-root-relative paths and
discovered sequence-level properties only; raw images and annotations remain untouched and ignored.

Keep images, annotations archives, extracted raw datasets, detections, model weights, and other large artifacts outside Git.
