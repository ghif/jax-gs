# 2D Gaussian Splatting (2DGS) Performance Optimization Strategy for TPU v6e

Based on a detailed analysis of the current JAX-based 2D Gaussian Splatting implementation, the training speed of 3.41 iterations per second on a 4x TPU v6e setup is heavily bottlenecked by single-device utilization and inefficient fallback rasterization. 

Below is a comprehensive strategy to resolve these bottlenecks and fully exploit the TPU v6e architecture.

## 1. Implement Data Parallelism (Multi-Device Scaling)
**Current Bottleneck:** The `train_truck.py` script executes the `train_step` function on a single camera image per iteration without utilizing JAX's distributed capabilities. On a machine with 4 TPU v6e cores, **75% of the compute hardware is sitting idle.**

**Action Plan:**
*   **Batch Cameras:** Instead of picking a single random camera `idx`, sample a batch of `num_devices` (e.g., 4) cameras per step.
*   **Replicate State:** Replicate the 2DGS parameters (`gaussians`) and the optimizer state across all 4 TPU devices using `flax.jax_utils.replicate` or JAX's `device_put_replicated`.
*   **Use `jax.pmap`:** Wrap the `train_step` with `jax.pmap` (with `axis_name='batch'`).
*   **Synchronize Gradients:** Inside the `pmap` function, after computing gradients with `jax.value_and_grad`, use `jax.lax.pmean(grads, axis_name='batch')` to average the gradients across the 4 devices before applying the optax update.
*   *Expected Impact:* A near 4x immediate increase in effective throughput (images processed per second).

## 2. Develop a 2DGS-Specific Pallas Kernel
**Current Bottleneck:** In `jax_gs/renderer/renderer.py`, if `mode == "2dgs"`, the `--use_pallas` flag is completely ignored, forcing the engine to fall back to `render_tiles_2d`. This pure JAX implementation heavily relies on `jax.vmap` and `jax.lax.scan`, which causes massive intermediate memory materialization and destroys hardware pipelining.
Furthermore, 2DGS requires depth, squared depth, normal maps, and accumulated weights for the Depth Distortion and Normal Consistency losses, which the current 3DGS Pallas kernel (`rasterize_kernel_tpu`) does not output (it only outputs RGBA).

**Action Plan:**
*   **Extend the TPU Pallas Kernel:** Duplicate and extend `rasterize_kernel_tpu` to `rasterize_2dgs_kernel_tpu`.
*   **Expand Accumulators:** Instead of just 4 channels (RGBA), the registers inside the kernel must accumulate 9 components per pixel:
    *   `Color` (3 channels)
    *   `Depth` (1 channel)
    *   `Depth Squared` (1 channel)
    *   `Normal` (3 channels)
    *   `Transmittance / Weight` (1 channel)
*   **Return Full Output:** Update the `out_specs` to return a `(H, W, 9)` tensor.
*   **Enable in Renderer:** Modify `renderer.py` to route `mode == "2dgs"` calls to this new Pallas kernel when `use_pallas=True`, avoiding the slow `render_tiles_2d` fallback.
*   *Expected Impact:* Orders of magnitude faster rasterization (potentially 10x+ faster forward/backward passes) due to utilizing TPU vector registers (`pl.ds` block loads) and avoiding `lax.scan`.

## 3. Optimize Tile Interaction Generation
**Current Bottleneck:** The `get_tile_interactions` function in `rasterizer.py` uses an $8 \times 8$ grid broadcast (`abs_x`, `abs_y`) to assign Gaussians to tiles. This creates an intermediate array of shape `[num_points, 64]`, which for 300,000+ points requires materializing nearly 20 million 32-bit integers in HBM. This causes high memory bandwidth pressure and slows down execution.

**Action Plan:**
*   **Reduce Memory Footprint:** Replace the broad `jnp.broadcast_to` logic with a custom Pallas kernel or `jax.vmap` over points that dynamically outputs sparse `(tile_id, gaussian_id)` pairs, filtering out invalid interactions *before* expanding memory.
*   **Radix Sort Tuning:** The bit-packed integer sort (`jax.lax.sort_key_val`) works well, but ensuring the input sizes are dynamically padded to a tighter upper bound (rather than a massive constant) will speed up the TPU sorting primitives.

## 4. JIT-Compile Loss Function Composition
**Current Bottleneck:** 2DGS evaluates multiple complex losses per step: L1, SSIM, Depth Distortion, and Normal Consistency. Currently, these are combined sequentially inside `loss_fn`. The SSIM implementation uses `jax.lax.conv_general_dilated` which is moderately fast but can be memory intensive if gradients are tracked naively.

**Action Plan:**
*   Since `train_step` is already `jax.jit` compiled, ensure that intermediate maps (like `extras["depth"]` and `extras["normals"]`) are strictly consumed and discarded immediately without forcing the TPU to hold them in HBM across the backward pass longer than necessary.
*   The `stop_gradient` applied in `depth_distortion_loss` and `normal_consistency_loss` is correct, but check that XLA is successfully fusing the gradient calculations with the rendering output nodes.

## Summary Roadmap

1. **Quick Win:** Implement `jax.pmap` over the 4 TPU v6e cores in `train_truck.py` to process batches of 4.
2. **High ROI:** Write a `rasterize_2dgs_kernel_tpu` in Pallas that outputs the 9 required channels and activate it in `renderer.py` for `mode="2dgs"`.
3. **Refinement:** Rewrite `get_tile_interactions` to minimize HBM footprint during the tile assignment broadcast.