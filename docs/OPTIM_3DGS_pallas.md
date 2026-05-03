# Strategy: Accelerating 3DGS Training with Custom Pallas Backward Kernels

## 1. Problem Statement
The current JAX-GS implementation utilizes **JAX Pallas** kernels to accelerate the forward rasterization pass on both TPU (Mosaic) and GPU (Triton). While this provides significant speedups for inference, training performance is limited because `pl.pallas_call` functions are opaque to JAX's automatic differentiation system. 

Currently, training either:
1.  **Fails** if `jax.grad` is called on a `pallas_call` without a custom VJP.
2.  **Falls back** to suboptimal pure JAX implementations that suffer from massive HBM memory traffic and redundant computations.

To achieve state-of-the-art training speeds on a single accelerator, we must implement a hand-optimized backward pass using Pallas.

## 2. Proposed Strategy: Full Custom VJP and Backward Kernels

We propose implementing a dedicated backward pass that mirrors the efficiency of the forward pass by performing gradient accumulation directly in on-chip memory (SRAM/VMEM).

### A. `jax.custom_vjp` Integration
We will wrap the `render_tiles_pallas` function in `jax_gs/renderer/rasterizer_pallas.py` with `jax.custom_vjp`.

*   **Forward Pass (`fwd`):** In addition to the rendered image, the forward pass will return a "residual" or "checkpoint" state. This state will include:
    *   `final_T`: The final transmittance for each pixel.
    *   `sorted_gaussian_ids`: The IDs of Gaussians contributing to each tile.
    *   `tile_boundaries`: Indices for efficient lookup in the backward pass.
*   **Backward Pass (`bwd`):** The backward function will receive the gradients of the loss with respect to the output image ($\partial L / \partial C$) and invoke a specialized Pallas backward kernel.

### B. Custom Backward Pallas Kernels
The core of the acceleration lies in the backward kernel, which reverses the forward alpha-blending process.

#### 1. GPU (Triton) Optimization
*   **Tile-Local Accumulation:** For each 16x16 tile, the kernel will load the $\partial L / \partial C$ gradients into SRAM.
*   **Reverse Iteration:** The kernel iterates through the sorted Gaussians in *reverse* order (from back to front) to maintain correct gradient bookkeeping.
*   **Atomic Aggregation:** Because multiple tiles might influence the same Gaussian, the kernel will use `atomic_add` to update the global gradient buffers for Gaussian means, scales, rotations, and opacities.

#### 2. TPU (Mosaic) Optimization
*   **VMEM Buffering:** Since TPUs lack efficient global hardware atomics, the kernel will write gradients to a tile-specific buffer in HBM.
*   **Vectorized Gradients:** The kernel will process 128-float or 256-float vectors to maximize MXU/VPU utilization during the gradient calculation.
*   **Two-Pass Reduction:**
    1.  **Kernel Pass:** Compute and store gradients per-tile per-Gaussian.
    2.  **JAX Pass:** Use `jax.lax.segment_sum` or a similar highly-optimized XLA primitive to aggregate these per-tile gradients into the final global gradient array.

### C. Checkpointing & Recomputation
To balance memory usage and speed:
*   **Don't save everything:** Saving the full intermediate weight of every Gaussian for every pixel is memory-prohibitive.
*   **Partial Recomputation:** We will save the final transmittance and recompute local alpha values inside the backward kernel using the original Gaussian parameters, which are already in memory.

## 3. Expected Impact
*   **Training Speedup:** 3x - 10x reduction in training step latency compared to pure JAX `vmap/scan` approaches.
*   **Memory Efficiency:** Drastic reduction in HBM traffic by keeping the most active gradient accumulation inside SRAM/VMEM.
*   **Scalability:** Provides a foundation for real-time 3DGS training on a single TPU v5p or A100/H100.

## 4. Implementation Roadmap
1.  **Phase 1:** Modify `rasterizer_pallas.py` to support `custom_vjp`.
2.  **Phase 2:** Implement the Triton backward kernel (GPU).
3.  **Phase 3:** Implement the Mosaic backward kernel (TPU).
4.  **Phase 4:** Benchmark training on the `fern` dataset and verify gradient correctness vs. pure JAX.
