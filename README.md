# JAX Gaussian Splatting

A minimal, high-performance, JAX-based implementation of 3D Gaussian Splatting. Restructured with a clean, modular architecture in the `jax_gs` package.

## Features

- **Clean Architecture**: Core logic modularized into `jax_gs` (core, renderer, io, training).
- **Optimized Tile Rasterizer**: JAX-native implementation with efficient bit-packed sorting for CPU, CUDA, and Apple Silicon (MPS).
- **Fast GPU Execution**: Optimized for NVIDIA L4 GPUs with full `float32` throughput.
- **Resume Training**: Continue training from any saved PLY checkpoint.
- **Unit Tested**: Comprehensive test suite for mathematical correctness and IO.

## Training Demo

![Training Results](results/progress_video.mp4)

## Environment Setup (using uv)

This project recommends using `uv` for fast Python package management.

1.  **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Setup Environment** (`.cpu_env`):
    ```bash
    uv venv .cpu_env --python 3.11
    source .cpu_env/bin/activate
    uv pip install -r requirements_cpu.txt
    ```

## Data Preparation

1.  Download the **Fern** dataset (from NeRF LLFF data).
2.  Place it in `data/nerf_example_data/nerf_llff_data/fern`.
3.  The directory should contain `images_8/` and `sparse/0/`.

## Usage

### Training

Start a new training session on the Fern dataset:
```bash
python train_fern.py --num_iterations 10000
```

### Resume Training

Continue training from the latest `.ply` checkpoint:
```bash
python train_fern_resume.py --num_iterations 5000
```

**Parameters:**
- `--num_iterations`: Total iterations for `train_fern.py` or *additional* iterations for `train_fern_resume.py`. Default is 10000.

**Outputs:**
- **Progress Images**: `results/fern_YYYYMMDD_HHMMSS/progress/`.
- **PLY Checkpoints**: `results/fern_YYYYMMDD_HHMMSS/ply/`.

### Visualization

Visualize trained splats using the Viser-based viewer:
```bash
python viewer_ply.py results/fern_YYYYMMDD_HHMMSS/ply/fern_final.ply
```

## Quality Assurance

### Run Unit Tests
To verify mathematical correctness and IO stability, run the test suite using `pytest`.

```bash
# Recommended: Run on CPU for deterministic numerical checks
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/
```

If you encounter environment issues, you can explicitly point to your virtual environment's site-packages:
```bash
JAX_PLATFORMS=cpu PYTHONPATH=.:$(pwd)/.cpu_env/lib/python3.11/site-packages pytest tests/
```

## Project Structure

- `jax_gs/`: Core package containing:
    - `core/`: `Gaussians` and `Camera` data structures.
    - `renderer/`: Tiled rasterization and projection kernels.
    - `io/`: COLMAP and PLY loading/saving logic.
    - `training/`: Loss functions and JIT-compiled trainer step.
- `tests/`: Unit tests for each module.
- `train_fern.py`: Entry point for training.
- `train_fern_resume.py`: Entry point for resuming training.
- `viewer_ply.py`: PLY visualization script.
