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
