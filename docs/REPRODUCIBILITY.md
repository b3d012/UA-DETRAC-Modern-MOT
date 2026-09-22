# Reproducibility Contract

Every experiment should record enough information to reconstruct the software and execution context.

Required fields:

- experiment ID;
- UTC timestamp;
- Git commit SHA and dirty/clean state;
- config file(s) and a stable config hash;
- seed(s);
- OS/platform;
- CPU model and logical/physical core count where available;
- GPU model(s), VRAM, and driver;
- CUDA runtime/toolkit information where available;
- Python version;
- PyTorch/torchvision version and CUDA availability;
- versions of core detector/tracker/evaluation libraries once introduced;
- command line used to launch the run.

The project should centralize this logic instead of duplicating environment-printing code across scripts.

## M1 environment capture

From an installed development environment, capture the current host and configuration state with:

```bash
python scripts/capture_environment.py
```

The default output is `outputs/environment/m1_environment.json`. It is intentionally ignored because
timestamps and hardware details are host-specific. Pass `--output` to select another generated path,
or repeat `--config` to hash a different explicit set of configuration files. Existing files are
preserved unless `--force` is supplied. Missing optional PyTorch, CUDA, or NVIDIA tooling is recorded
as unavailable rather than treated as an error.
