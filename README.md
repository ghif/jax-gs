# JAX-GS: High-Performance 3D & 2D Gaussian Splatting in Pure JAX

JAX-GS is a fast, hardware-agnostic implementation of 3D and 2D Gaussian Splatting entirely written in JAX. It's designed to saturate Matrix Multiply Units (MXUs) on Google TPUs and Tensor Cores on GPUs by compiling massive training loops into single execution graphs.

## Features
- **Pure JAX Rasterizer**: No custom CUDA kernels required. Runs efficiently on TPUs, GPUs, and CPUs.
- **Hardware Agnostic Acceleration**: Supports single-device execution (`jax.jit`) and multi-device data parallelism (`jax.pmap`) for linear scaling.
- **TPU Saturation**: The entire training loop and data sampling happen on-device, removing Python dispatch overhead and host-device bottlenecks.
- **2D & 3D Splatting**: Includes parallel modules for standard 3DGS as well as 2DGS (Surfel) representations.
- **Asynchronous I/O**: Training never stops to wait for disk operations (PLY checkpointing and image saving happen in the background).

---

## 🚀 Getting Started (Tutorial)

Follow this step-by-step tutorial to setup your environment and train your first 3D Gaussian Splatting model.

### 1. Environment Setup

We recommend using `uv` for fast Python environment management. 

#### For TPU environments:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .tpu_env --python 3.11
source .tpu_env/bin/activate
uv pip install -r requirements_tpu.txt
```

#### For CPU/GPU environments:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .cpu_env --python 3.11
source .cpu_env/bin/activate
uv pip install -r requirements_cpu.txt
```

### 2. Dataset Preparation

JAX-GS natively expects standard COLMAP-formatted datasets (such as the NeRF LLFF datasets). 

1. Download the **Fern** dataset (or use your own custom COLMAP output).
2. Place it in `data/nerf_example_data/nerf_llff_data/fern`.
3. Ensure the directory contains an `images/` (or `images_8/`) folder and the COLMAP `sparse/0/` folder.

### 3. Training on a Single Device

To train a model on a single GPU or TPU core, use `train.py`. The script compiles training blocks (e.g., 100-500 steps) to run entirely on the accelerator, making it blazing fast.

```bash
python train.py --data_path data/nerf_example_data/nerf_llff_data/fern \
                --images_subdir images_8 \
                --num_iterations 10000 \
                --fast_tpu_rasterizer
```
*(Note: Omit `--fast_tpu_rasterizer` if you are not running on a Google TPU)*

**Outputs will be automatically saved to a timestamped directory in `results/`:** (e.g., `results/fern_3dgs_fast_tpu_YYYYMMDD_HHMMSS/`)
- Progress renderings: `progress/`
- Checkpoints: `ply/`

### 4. Training on Multiple Devices (Data Parallelism)

If you have multiple TPU cores (e.g., a TPU v4-8 or v5e pod) or a multi-GPU instance, you can run `train_parallel.py`. This uses `jax.pmap` to replicate the model across devices, processing multiple views concurrently to effectively multiply your batch size.

```bash
python train_parallel.py --data_path data/nerf_example_data/nerf_llff_data/fern \
                         --images_subdir images_8 \
                         --num_iterations 10000 \
                         --fast_tpu_rasterizer
```

### 5. Resuming Training

If your training was interrupted or you want to continue optimizing an existing scene, you can resume using `train_fern_resume.py`. This script automatically finds the latest `.ply` checkpoint generated in your `results/` directory and continues training.

```bash
python train_fern_resume.py --num_iterations 5000
```
*(This adds 5000 iterations to the loaded checkpoint)*

### 6. Visualizing the Results

We include real-time, interactive web viewers built with `viser`. 

To visualize your trained splat checkpoint:
```bash
python viewer_ply.py results/fern_3dgs_.../ply/step_...ply
```
*Navigate to `http://localhost:8080` in your browser to view and navigate the 3D scene.*

To inspect randomly initialized geometry (helpful for understanding the underlying data structures), run:
```bash
python viewer_random.py --num 5000
```

---

## 🛠️ Project Architecture

JAX-GS is architected cleanly with isolated responsibilities, supporting both 3D and 2D Gaussian models:

- `jax_gs/`: Core 3DGS implementation
  - `core/`: State and mathematical definitions (`Gaussians`, `Camera`).
  - `renderer/`: Tiled rasterizer, bit-packed sorting, and alpha blending in pure JAX.
  - `training/`: Loss calculation, JIT-compatible adaptive density control, and optimizer steps.
  - `io/`: Utilities for PLY and COLMAP reading/writing.
- `jax_2dgs/`: 2DGS (Surfel) implementation with identical modularity.
- `docs/`: Technical blogs and architecture details on TPU saturation strategies, optimizations, and benchmarks.

## 🧪 Testing and Quality Assurance

We use `pytest` for unit testing to ensure mathematical correctness and I/O stability across platforms. For deterministic numerical checks, we recommend running the tests on the CPU.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/
```

If you encounter Python path issues, you can explicitly point to the `site-packages` of your active virtual environment:
```bash
JAX_PLATFORMS=cpu PYTHONPATH=.:$(pwd)/.tpu_env/lib/python3.11/site-packages pytest tests/
```