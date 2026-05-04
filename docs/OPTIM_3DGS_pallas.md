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

#### 2. TPU (Mosaic) Optimization (Revised)
Based on empirical benchmarking, custom Mosaic TPU kernels for 3DGS rasterization are extremely brittle and suffer from severe performance bottlenecks (e.g., 0.02x speedup vs. standard JAX) due to strict constraints on memory layouts, 128-element alignment, and unsupported vector.multi_reduction operations over custom block shapes.

The new strategy for TPU optimization shifts away from writing manual Mosaic kernels to leveraging XLA's highly optimized compiler on structured JAX code:

*   **Abandon Hand-Written Mosaic Kernels:** Remove the failing rasterize_kernel_tpu and rasterize_bwd_kernel_tpu implementations. They are opaque to the XLA compiler's layout optimization passes, resulting in unoptimized memory access patterns and compilation crashes.
*   **Fully Vectorized Pure JAX:** Rewrite the core rasterization logic using standard XLA-friendly JAX primitives. XLA on TPU excels at compiling large blocks of vmap, scan, and matmul operations.
*   **Tile-Based Parallelism via vmap and scan:** 
    *   Pre-sort and chunk Gaussians per tile.
    *   Use jax.vmap over tiles and jax.lax.scan over the chunked Gaussians within each tile. XLA can fuse these loops and map them efficiently to the TPU's matrix multiplier units (MXUs) and vector processing units (VPUs) without manual block layout specification.
*   **Avoid Dynamic Indexing in Inner Loops:** The previous Pallas implementations relied heavily on manual gather/scatter indexing inside the kernel. The revised pure JAX approach must use fixed-size arrays and padding to ensure XLA can statically prove memory alignment, avoiding XLA layout alignment errors observed previously.

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
