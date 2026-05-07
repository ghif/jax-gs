# JAX-GS

Pure-JAX 3D Gaussian Splatting with tiled rasterization, static-shape adaptive density control, and single-device or multi-device training paths.

The repository contains:

- `jax_gs/`: 3D Gaussian Splatting implementation.
- `jax_2dgs/`: 2DGS / surfel experiments.
- `train.py`: single-device 3DGS training.
- `train_parallel.py`: multi-device data-parallel 3DGS training with `jax.pmap`.
- `viewer_ply.py`: interactive viewer for saved `.ply` checkpoints.
- `viewer_random.py`: viewer for randomly generated Gaussians.
- `tests/`: regression and unit tests.

## Visual Results

### Training Progress
These animations show the evolution of the Gaussian Splats during the optimization process.

![Fern Training](results/fern_training.gif)

### Rendered Views
These are front-facing "wiggle" views generated from the final `.ply` checkpoints.

| Fern (View) | Room (View) |
| :---: | :---: |
| ![Fern View](results/fern_viewing.gif) | ![Room View](results/room_viewing.gif) |

## What Is Current In This Version

The current codebase is centered on the newer JAX-friendly implementation:

- Pure JAX renderer and rasterizer. No custom CUDA kernels.
- Static-capacity `DensityState` with `active_mask` for JIT-compatible densification.
- Adaptive clone / split / prune with optimizer-state reordering.
- Renderer health metrics used to gate densification and SH promotion.
- Standard rasterizer plus TPU-optimized rasterizer via `--fast_tpu_rasterizer`.
- Single-device and multi-device training paths using the same core model.

## Requirements

- Python 3.12.x recommended.
- Conda environment recommended for local development.
- A COLMAP-format dataset for training.

For local work in this repository, the intended environment is:

```bash
conda activate tpu-env
python --version
```

Install dependencies:

```bash
pip install -r requirements_tpu.txt
```

If you are setting up a separate CPU-only environment, use `requirements_cpu.txt` instead.

## Cloud TPU VM Tutorial

This section covers how to provision, connect, and set up a Google Cloud TPU VM for training.

### 1. Provision a TPU VM (Flex-start)

The most cost-effective way to get high-end TPUs (like Trillium v6e) is using the `flex-start` provisioning model.

```bash
# Example for a v6e-4 (Trillium) slice
gcloud alpha compute tpus queued-resources create jax-gs-queue \
    --zone=southamerica-east1-c \
    --accelerator-type=v6e-4 \
    --runtime-version=v2-alpha-tpuv6e \
    --node-id=my-tpu-node \
    --provisioning-model=flex-start
```

Check the status of your request:
```bash
gcloud alpha compute tpus queued-resources describe jax-gs-queue --zone=southamerica-east1-c
```

Once the state is `ACTIVE`, the TPU VM is ready.

### 2. Connect via SSH

Use port forwarding (e.g., `8080`) if you intend to use the interactive PLY viewer on the TPU VM.

```bash
gcloud compute tpus tpu-vm ssh my-tpu-node \
    --zone=southamerica-east1-c \
    --ssh-flag="-L 8080:localhost:8080"
```

### 3. Machine Setup

Once logged into the TPU VM, set up the environment:

```bash
# Install Miniforge (Conda)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc

# Clone and enter the repo
git clone https://github.com/ghif/jax-gs
cd jax-gs

# Create and activate environment
conda create -n tpu-env python=3.12 -y
conda activate tpu-env

# Install JAX with TPU support
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install project requirements
pip install -r requirements_tpu.txt
```

### 4. Verify TPU Access

```bash
python -c "import jax; print(jax.devices())"
```
You should see one or more `TpuDevice` entries.



## Dataset Layout

The training scripts expect a COLMAP-style scene directory:

```text
data/nerf_example_data/nerf_llff_data/fern/
├── images_8/          # or images/
└── sparse/
    └── 0/
```

The loader is implemented in `jax_gs/io/colmap.py` and is called by both `train.py` and `train_parallel.py`.

## Quick Start

### 1. Single-device training

Use `train.py` for a single GPU, single TPU core, or CPU-based debugging.

```bash
python train.py \
  --data_path data/nerf_example_data/nerf_llff_data/fern \
  --output_path results \
  --images_subdir images_8 \
  --num_iterations 10000
```

If you are training on TPU, enable the TPU rasterizer:

```bash
python train.py \
  --data_path data/nerf_example_data/nerf_llff_data/fern \
  --output_path results \
  --images_subdir images_8 \
  --num_iterations 10000 \
  --fast_tpu_rasterizer
```

### 2. Multi-device training

Use `train_parallel.py` when multiple JAX devices are visible.

```bash
python train_parallel.py \
  --data_path data/nerf_example_data/nerf_llff_data/fern \
  --output_path results \
  --images_subdir images_8 \
  --num_iterations 10000 \
  --fast_tpu_rasterizer
```

This path replicates model state across devices, averages gradients with `pmean`, and performs densification on an authoritative unreplicated state before broadcasting it back to replicas.

### 3. View a trained checkpoint

The trainers save progress images under `progress/` and write a rolling checkpoint at:

```text
results/<run_name>/ply/<scene_name>_latest.ply
```

They also save a final checkpoint:

```text
results/<run_name>/<scene_name>_final.ply
```

To inspect a saved checkpoint:

```bash
python viewer_ply.py results/<run_name>/ply/<scene_name>_latest.ply
```

The viewer starts on port `8080` by default.

### 4. Visualize random Gaussians

```bash
python viewer_random.py --num 2000
```

## Training Flags That Matter

Both `train.py` and `train_parallel.py` expose the main controls you will likely tune first:

- `--fast_tpu_rasterizer`: enables the TPU-specialized rasterizer path.
- `--images_subdir`: choose `images`, `images_2`, `images_4`, `images_8`, etc.
- `--max_gaussians_cap`: hard upper bound for padded Gaussian capacity.
- `--max_gaussians_growth`: growth multiplier relative to the initial COLMAP point count.
- `--density_interval`: frequency of densify / prune passes during the density window.
- `--max_overflow_tiles`: hold densification / SH promotion if too many tiles spill beyond a single chunk.
- `--max_overflow_interactions`: same idea, measured by overflow interaction count.
- `--max_radius_cap_violations`: hold growth when too many splats exceed the tile-span budget.
- `--max_truncated_tiles`: threshold for hard truncation in the rasterizer.
- `--max_truncated_interactions`: threshold for truncated interaction count.
- `--sh_promotion_mode`: `health_gated` or `always`.

Example with the main stability-related flags made explicit:

```bash
python train.py \
  --data_path data/nerf_example_data/nerf_llff_data/fern \
  --output_path results \
  --images_subdir images_8 \
  --num_iterations 30000 \
  --fast_tpu_rasterizer \
  --density_interval 500 \
  --max_gaussians_growth 8 \
  --max_gaussians_cap 200000 \
  --sh_promotion_mode health_gated
```

## Outputs

A local run with `--output_path results` creates a directory like:

```text
results/
└── fern_3dgs_fast_tpu_YYYYMMDD_HHMMSS/
    ├── progress/
    │   ├── progress_1000.png
    │   ├── progress_2000.png
    │   └── ...
    ├── ply/
    │   └── fern_latest.ply
    └── fern_final.ply
```

The rolling `.ply` is intentionally overwritten to keep artifact size manageable.

## Testing

Run tests on CPU unless you are validating accelerator-specific behavior:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu pytest tests/
```

`pytest.ini` currently skips:

- `tests/test_benchmark_3dgs_vs_2dgs.py`
- `tests/test_gaussians_2d.py`

Run those explicitly if you need them.

## Repository Layout

### Core packages

- `jax_gs/core/`: camera and Gaussian parameter definitions.
- `jax_gs/renderer/`: projection, SH evaluation, tile interaction building, rasterization.
- `jax_gs/training/`: losses, train step logic, adaptive density control.
- `jax_gs/io/`: COLMAP and PLY I/O.

### Scripts

- `train.py`: recommended single-device training path.
- `train_parallel.py`: recommended multi-device training path.
- `viewer_ply.py`: interactive viewer for saved checkpoints.
- `viewer_random.py`: interactive viewer for synthetic Gaussian clouds.
- `train_fern_resume.py`: older script kept in the repo, not aligned with the current main training pipeline.

### Documentation

- `ARCHITECTURE.md`: architectural overview.
- `docs/BLOG_EFFICIENT_3DGS.md`: implementation note on the current JAX 3DGS pipeline.
- `docs/`: additional benchmark and design notes.

## Development Notes

- Keep JAX code vectorized and static-shape friendly where possible.
- Prefer CPU test runs for deterministic debugging.
- Generated outputs belong in `results/` or `results_test/` and should not be committed.

## Common Problems

### Training writes to `gs://...` instead of a local directory

Pass `--output_path results` explicitly. The scripts default to a GCS path.

### TPU rasterizer on non-TPU hardware

Do not pass `--fast_tpu_rasterizer` unless you actually want the TPU-optimized path.

### Dataset not found

Double-check:

- `--data_path`
- `--images_subdir`
- presence of `sparse/0/` under the scene directory

### No multi-device scaling

Check what JAX sees:

```bash
python -c "import jax; print(jax.devices())"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
