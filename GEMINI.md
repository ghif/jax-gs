# GEMINI.md - JAX-GS Project Context

## Project Overview
JAX-GS is a high-performance, pure-JAX implementation of 3D and 2D Gaussian Splatting (3DGS/2DGS). It leverages JAX's composable transformations (`jit`, `vmap`, `pmap`, `lax.scan`) and the XLA compiler to achieve hardware-agnostic acceleration, specifically optimized for Google TPUs and NVIDIA GPUs.

### Key Technologies
- **JAX:** Core framework for differentiation and hardware acceleration.
- **Optax:** Gradient processing and optimization library.
- **Chex:** Strongly-typed dataclasses and utilities.
- **COLMAP:** Standard dataset format for 3D reconstruction.

## Architecture
The project is modularized into several core packages:
- `jax_gs/core/`: Parameter definitions for Gaussians (`gaussians.py`) and Cameras (`camera.py`).
- `jax_gs/renderer/`: Mathematical rendering engine, including projection (`projection.py`) and rasterization (`rasterizer.py`, `rasterizer_tpu.py`).
- `jax_gs/training/`: Training orchestration, losses (`losses.py`), and adaptive density control (`density.py`, `trainer.py`).
- `jax_gs/io/`: I/O for COLMAP (`colmap.py`) and PLY (`ply.py`) formats.

## Development Conventions

### JAX & Performance
- **Pure JAX:** All rendering logic must be expressed as pure JAX array operations. Avoid custom CUDA kernels to maintain TPU compatibility.
- **Static Shapes & JIT:** Training loops and densification logic use static-shape `DensityState` with an `active_mask`. This allows the entire training loop to be JIT-compiled.
- **On-Device Loops:** Use `jax.lax.scan` to compile blocks of training steps into a single XLA graph, minimizing host-device dispatch overhead.
- **Memory Optimization:** Use `jax.checkpoint` in the renderer (especially in `rasterizer_tpu.py`) to trade compute for memory via rematerialization.

### State Management
- **Gaussian State:** Represented by `jax_gs.core.gaussians.Gaussians` (a `chex.dataclass`).
- **Trainer State:** Managed via `jax_gs.training.density.DensityState`, which tracks Gaussian parameters, optimizer state, `active_mask`, and gradient accumulators.
- **Masking:** Always mask updates using `active_mask` to ensure inactive/padded Gaussians are not modified during training.

### Multi-Device Scaling
- **SPMD:** Use `jax.pmap` for multi-device data parallelism.
- **Synchronization:** Use `jax.lax.pmean` to average gradients and metrics across devices within the `pmap` block.

## Building and Running

### Environment Setup
The project uses a specific conda environment:
```bash
conda activate tpu-env
pip install -r requirements_tpu.txt
```

### Training
- **Single-Device:**
  ```bash
  python train.py --data_path <DATA_DIR> --output_path results --images_subdir images_8
  ```
- **Multi-Device (Parallel):**
  ```bash
  python train_parallel.py --data_path <DATA_DIR> --output_path results --images_subdir images_8
  ```
- **TPU Optimization:** Add `--fast_tpu_rasterizer` when training on TPU.

### Verification and Testing
Run tests using `pytest`. It is recommended to use the CPU platform for deterministic debugging:
```bash
PYTHONPATH=. JAX_PLATFORMS=cpu pytest tests/
```

### Checkpoint Inspection
Checkpoints are saved as `.ply` files. Use the interactive viewer:
```bash
python viewer_ply.py results/<run_name>/ply/<scene_name>_latest.ply
```

## Project Memory (Private)
User-specific notes and local workflow details are stored in `/home/mghifary/.gemini/tmp/jax-gs/memory/MEMORY.md`. Always check this file for local environment overrides (e.g., active branch, specific conda commands).
