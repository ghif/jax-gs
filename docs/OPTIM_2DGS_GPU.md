# Deep Optimization Strategy for 2DGS/3DGS Training on GPU A100 (JAX)

## 1. Eliminate Host-to-Device (CPU) Bottlenecks
**Issue:** The `train_truck.py` script has a python `for i in pbar:` loop that samples random camera indices `idx = random.randint(...)` and passes python objects (`target = jax_targets[idx]`, `cam.W2C`) to `train_step`. This forces JAX to dispatch from Python to C++ every iteration, preventing the GPU from being fully saturated.
**Optimization:** 
- **Pre-load onto Device:** Stack all `jax_targets` and `jax_cameras` into single device arrays `(N, H, W, C)` upfront using `jax.device_put`. 
- **Device-Side Sampling:** Instead of passing single frames from Python, pass a `jax.random.PRNGKey` to `train_step` and use `jax.random.choice` to select the training view *on the device*. This allows you to compile multiple training steps into a single kernel using `jax.lax.scan` or `jax.lax.fori_loop`, drastically reducing host overhead.

## 2. Leverage A100 Tensor Cores & Mixed Precision
**Issue:** Standard training defaults to FP32, which wastes the massive BF16/FP16 bandwidth available on the A100. Memory bandwidth is typically the bottleneck in Gaussian Splatting rasterization.
**Optimization:**
- Store Spherical Harmonics (SH) coefficients, opacity, and base colors in `jnp.bfloat16`. 
- Only keep critical geometry parameters (like 3D means and 2D projections) and the accumulator buffers in `fp32`.
- Compile XLA with `XLA_FLAGS="--xla_gpu_triton_gemm_any=True"` to maximize Tensor Core usage for any implicit matrix multiplications.

## 3. Pre-allocated Buffers for Adaptive Density Control
**Issue:** In 2D/3DGS, Gaussians are dynamically cloned, split, and pruned. In JAX, dynamically changing array shapes triggers an expensive XLA recompilation. If `train_step` creates arrays of new sizes, the A100 will stall constantly.
**Optimization:**
- Allocate a **fixed-size buffer** equal to the maximum expected number of Gaussians (e.g., 5-10 million).
- Maintain an `is_active` boolean mask array.
- Use `jax.lax.cond` or `jax.numpy.where` to handle logic based on the mask.
- Pass the padded state arrays to the rasterizer, which can aggressively skip inactive Gaussians using early-exit logic in the Pallas/Triton kernels.

## 4. Maximize Memory Reuse via In-Place Updates
**Issue:** `state = (gaussians, opt_state)` is updated at each step. In naive JAX, this creates a new copy of the arrays, leading to OOM or high GC overhead on large scenes.
**Optimization:**
- Annotate `train_step` with `@jax.jit(donate_argnums=(0,))` so XLA knows it can safely overwrite the `state` arrays in-place in VRAM. This is critical for A100 to avoid buffer fragmentation.

## 5. Optimized Pallas Custom Kernel (Tile-Based Rasterization)
**Issue:** A naive scatter/gather approach to rendering in JAX performs poorly on GPUs.
**Optimization:** 
- Ensure `use_pallas=True` routes to a heavily optimized Tile-Based Rasterization kernel.
- The A100 has 40MB of L2 Cache and substantial shared memory (164 KB per SM). The Pallas kernel should partition the image into 16x16 tiles, load the tile metadata into Shared Memory, and process sorted Gaussians iteratively. 
- Use Radix Sort (`jax.lax.sort` or a custom Triton implementation) on 64-bit keys (32-bit depth + 32-bit tile ID) to minimize memory collisions.

## 6. Multi-GPU Data Parallelism (Pmap/Vmap)
**Issue:** For multi-device setups, the script uses `NamedSharding`. 
**Optimization:**
- Since A100 has massive compute, you can parallelize by duplicating the Gaussian parameters across all GPUs using `jax.pmap` (or SPDM `vmap` + `pjit`), computing gradients for different cameras locally on each GPU, and running `jax.lax.pmean` to average the gradients before applying the Optax update.
- With NVLink on A100 clusters, this all-reduce step is extremely fast and effectively multiplies your batch size by the number of GPUs.