# JAX Gaussian Splatting

A minimal, JAX-based implementation of 3D Gaussian Splatting. This repository includes scripts to train a model on the standard Fern dataset and visualize the results.

## Environment Setup (using uv)

This project recommendeds using `uv` for fast Python package management. We suggest creating separate environments for CPU and MPS (Metal Performance Shaders) execution to handle dependency differences.

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Setup CPU Environment** (`.cpu_env`):
    Use this for standard execution or if you don't have a Mac with Apple Silicon.
    ```bash
    # Create virtual environment (Python 3.11 recommended)
    uv venv .cpu_env --python 3.11

    # Activate
    source .cpu_env/bin/activate

    # Install dependencies
    uv pip install -r requirements_cpu.txt
    ```

3.  **Setup MPS (Mac GPU) Environment** (`.mps_env`):
    Use this for accelerated training on macOS (Apple Silicon).
    ```bash
    # Create virtual environment
    uv venv .mps_env --python 3.11

    # Activate
    source .mps_env/bin/activate

    # Install dependencies
    uv pip install -r requirements_mps.txt
    ```

    *Note: MPS support requires explicit `float32` enforcing which is handled by the `train_fern_mps.py` script.*


## Data Preparation

1.  Download the **Fern** dataset (from the NeRF LLFF data).
2.  Place it in the `data` directory so that the structure looks like this:
    ```text
    data/
    └── nerf_example_data/
        └── nerf_llff_data/
            └── fern/
                ├── images_8/   # Downsampled images
                └── sparse/     # COLMAP data
                    └── 0/
    ```

## Training

### CPU Training
To start training on the CPU (using `.cpu_env`):
```bash
source .cpu_env/bin/activate
python train_fern.py
```

### MPS (Mac GPU) Training
To start training on Mac GPU (using `.mps_env`):
```bash
source .mps_env/bin/activate
python train_fern_mps.py
```

This will:
*   Load the COLMAP data and images.
*   Initialize 3D Gaussians from the sparse point cloud.
*   Train for 10,000 iterations.
*   Save outputs to the `results/` directory.

**Outputs:**
*   **Progress Images**: `results/fern_YYYYMMDD_HHMMSS/progress/` (rendered views during training).
*   **PLY Checkpoints**: `results/fern_YYYYMMDD_HHMMSS/ply/` (saved splat files).

## Visualization

You can visualize the trained Gaussian Splats using the included Viser-based viewer.

Run the viewer with the path to a generated `.ply` file:

```bash
# Works in either environment
python viewer_ply.py results/fern_YYYYMMDD_HHMMSS/ply/fern_final_splats.ply
```

*(Replace `fern_YYYYMMDD_HHMMSS` with the actual timestamp folder generated during training)*

**Controls:**
*   **Left Click + Drag**: Rotate camera.
*   **Right Click + Drag**: Pan camera.
*   **Scroll**: Zoom.
*   The viewer runs in the browser (usually at `http://localhost:8080`).

## Project Structure

*   `train_fern.py`: Main training script (CPU).
*   `train_fern_mps.py`: Training script optimized for MPS (Mac GPU).
*   `renderer_v2.py`: Tile-based differentiable renderer (JAX).
*   `renderer_v2_mps.py`: MPS-optimized renderer with float32 enforcement and stable sorting.
*   `viewer_ply.py`: Script to load and visualize `.ply` files.
*   `gaussians.py`: Data structure and initialization for 3D Gaussians.
*   `requirements_cpu.txt`: Dependencies for CPU environment.
*   `requirements_mps.txt`: Dependencies for MPS environment.
