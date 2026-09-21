# Dataset Policy

## Non-negotiable split discipline

UA-DETRAC's official training and testing partitions are benchmark assets, not suggestions.

- Official train: 60 sequences.
- Official test: 40 sequences.
- The official test set must remain evaluation-only.
- Validation is carved out exclusively from the 60 training sequences.
- Splitting is performed at **sequence level**, never by individual frame, to avoid temporal leakage.
- No threshold selection, ablation choice, checkpoint selection, early stopping decision, detector confidence setting, NMS setting, association threshold, or tracker hyperparameter may be selected using official-test performance.

## Raw data

Raw frames and annotations are never committed to Git. Store them outside the repository and reference them with `UA_DETRAC_ROOT` or an equivalent explicit config override.

## Test-set access logging

M1 should make test-set evaluation an explicit command and record each test evaluation under an experiment ID. Exploratory scripts should default to train/validation data and require a deliberate flag to read official-test labels.
