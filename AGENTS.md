# Repository Guidelines

## Project Structure & Module Organization
`jax_gs/` contains the 3D Gaussian Splatting implementation, split into `core/`, `renderer/`, `training/`, and `io/`. `jax_2dgs/` mirrors that layout for 2DGS/surfel experiments. Top-level scripts such as `train.py`, `train_parallel.py`, `train_fern_resume.py`, `viewer_ply.py`, and `viewer_random.py` are the main entry points. Tests live in `tests/`, with a few ad hoc root-level checks like `test_densify.py`. Design notes and benchmarks are in `docs/` and `ARCHITECTURE.md`. Generated outputs go to `results/` or `results_test/` and should not be committed.

## Build, Test, and Development Commands
Use the shared Conda environment for local work:

```bash
conda activate tpu-env
python --version  # expected: Python 3.12.x
pip install -r requirements_tpu.txt
```

Use `requirements_cpu.txt` only when setting up a separate CPU-only environment. Typical commands:

```bash
python train.py --data_path data/nerf_example_data/nerf_llff_data/fern --images_subdir images_8
python train_parallel.py --data_path data/nerf_example_data/nerf_llff_data/fern --images_subdir images_8 --fast_tpu_rasterizer
python viewer_ply.py results/<run>/ply/<checkpoint>.ply
PYTHONPATH=. JAX_PLATFORMS=cpu pytest tests/
```

Run tests on CPU unless you are validating accelerator-specific behavior.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for dataclasses like `Gaussians` and `Camera`. Keep JAX code vectorized and side-effect free so it remains `jit`/`pmap` friendly. Prefer small, focused modules under the existing package splits instead of adding more top-level scripts. There is no configured formatter in the repo today, so match surrounding style closely and keep imports straightforward.

## Testing Guidelines
Use `pytest` and name files `test_*.py`. Add unit tests next to the affected subsystem under `tests/` and prefer deterministic numeric assertions. `pytest.ini` skips `tests/test_benchmark_3dgs_vs_2dgs.py` and `tests/test_gaussians_2d.py` by default, so run benchmark or experimental files explicitly when needed. For training or renderer changes, include at least one regression-oriented test or a reproducible command.

## Commit & Pull Request Guidelines
Recent history uses short conventional prefixes such as `fix:`, `feat:`, `docs:`, and `refactor:`; follow that pattern and keep the subject line specific. PRs should explain the affected path or script, note CPU/GPU/TPU validation, and include key commands used for verification. For viewer or rendering changes, attach screenshots or rendered outputs; for training changes, include the dataset, flags, and a brief summary of convergence or performance impact.
