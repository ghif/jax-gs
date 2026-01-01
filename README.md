# JAX Gaussian Splatting

A minimal, JAX-based implementation of 3D Gaussian Splatting. This repository includes scripts to train a model on the standard Fern dataset and visualize the results.

## Prerequisites

1.  **Python Environment**: Ensure you have Python 3.10+ installed.
2.  **Dependencies**: Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This implementation is optimized for CPU execution. JAX Metal (GPU) support on macOS is currently experimental and disabled by default due to version incompatibilities.*

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

To start training the Gaussian Splatting model on the Fern dataset:

```bash
python train_fern.py
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
python viewer_ply.py results/fern_YYYYMMDD_HHMMSS/ply/fern_final_splats.ply
```

*(Replace `fern_YYYYMMDD_HHMMSS` with the actual timestamp folder generated during training)*

**Controls:**
*   **Left Click + Drag**: Rotate camera.
*   **Right Click + Drag**: Pan camera.
*   **Scroll**: Zoom.
*   The viewer runs in the browser (usually at `http://localhost:8080`).

## Project Structure

*   `train_fern.py`: Main training script.
*   `renderer_v2.py`: Tile-based differentiable renderer implemented in JAX.
*   `viewer_ply.py`: Script to load and visualize `.ply` files.
*   `gaussians.py`: Data structure and initialization for 3D Gaussians.
*   `requirements.txt`: Python dependencies.
