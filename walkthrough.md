# Walkthrough - JAX 3D Gaussian Splatting (CPU Optimized)

This project implements a "pure JAX" 3D Gaussian Splatting system. The implementation is designed to be fully differentiable and runs reliably on CPU to ensure compatibility and correctness, bypassing experimental Metal limitations.

## Accomplishments

- **[NEW] [gaussians.py](file:///Users/mghifary/Work/Code/AI/jax-gs/gaussians.py)**: Implemented 3D Gaussian data structures and covariance computations.
- **[NEW] [renderer.py](file:///Users/mghifary/Work/Code/AI/jax-gs/renderer.py)**: Developed a differentiable rasterizer using `jax.lax.scan` and `jax.vmap`.
- **[NEW] [viewer.py](file:///Users/mghifary/Work/Code/AI/jax-gs/viewer.py)**: Integrated the `viser` library for interactive 3D visualization.
- **[NEW] [train.py](file:///Users/mghifary/Work/Code/AI/jax-gs/train.py)**: Set up a training loop with `optax` for optimizing Gaussian parameters.

## Verification Results

### CPU Execution (Primary)
The implementation is verified to work correctly on CPU:
- **Forward pass**: Successfully renders splats into images.
- **Backward pass**: Gradients are correctly computed through the rasterizer, enabling model optimization.
- **Device Isolation**: Successfully removed `jax-metal` and forced CPU-only mode to ensure stability on Mac.

## How to Run

### Training
To run the training loop with synthetic data:
```bash
conda activate jax-gs
python train.py
```

### Interactive Viewer
To launch the `viser` viewer:
```bash
conda activate jax-gs
python viewer.py
```
After launching, open the URL (usually `http://localhost:8081`) in your browser.