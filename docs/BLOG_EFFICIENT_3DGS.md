# Efficient 3D Gaussian Splatting in JAX-GS

## Introduction

3D Gaussian Splatting (3DGS) has emerged as a state-of-the-art technique for real-time radiance field rendering, offering a compelling alternative to neural volume integration methods like NeRF. In a [previous attempt](https://www.notion.so/Beyond-the-Triangle-Building-a-3D-Gaussian-Splatting-Engine-from-Scratch-in-JAX-2dca59d37360803b9ad8ec11ab117758?pvs=21), I implemented the 3DGS algorithm purely in JAX. However, that version was a naive approach and did not fully harness JAX’s performance on accelerators. There is still plenty of room to improve runtime.

This article discusses how to further optimize 3DGS across multiple Tensor Processing Units (TPUs). In general, the strategy involves restructuring the rasterization implementation and exploiting batched data parallelism. The `jax-gs` project addresses these challenges by reformulating 3DGS within the JAX framework and leveraging XLA (Accelerated Linear Algebra) to compile the entire training and rendering pipeline into highly optimized machine code. This transition from a dynamic, CUDA-centric model to a static-shape, JIT-compiled architecture allows `jax-gs` to exploit the massive parallel processing power of TPUs while maintaining numerical stability and structural consistency.

Unlike the original implementation by Kerbl et al. (2023), which relies on hand-crafted CUDA kernels and dynamic memory management in PyTorch, `jax-gs` is expressed through high-level JAX primitives. While the original 3DGS achieves high performance through specialized hardware-level sorting and rasterization on NVIDIA GPUs, our approach reformulates these operations as vectorized tensor manipulations. By leveraging XLA's ability to fuse and optimize complex graph operations, we achieve hardware-agnostic acceleration that scales seamlessly across TPU arrays. This shift allows for a research-friendly codebase that benefits from JAX's composable transformations (e.g., `vmap`, `pmap`, `grad`) while delivering the performance required for production-scale radiance field optimization.

## About Tensor Processing Units (TPUs)

TPUs are Google's custom-developed application-specific integrated circuits (ASICs) designed specifically to accelerate machine learning workloads. Unlike general-purpose GPUs, TPUs are architected around the requirements of deep learning, prioritizing high-throughput matrix multiplications and low-latency interconnects.

### TPU Evolution

Since their introduction, TPUs have undergone several generations of architectural refinement:
- **TPU v2/v3**: Focused on training large-scale models with significant memory bandwidth improvements.
- **TPU v4**: Introduced a 3D torus topology for superior scaling and doubled the performance per watt.
- **TPU v5e/v5p**: Optimized for cost-efficiency (v5e) and maximum performance (v5p) for LLM training and inference.
- **TPU v6e (Trillium)**: The latest generation, designed for the most demanding multi-modal and generative AI tasks, offering significant performance gains over its predecessors.

### TPU Chips, Pods, and Slices

The TPU architecture is modular and scalable:
- **TPU Chip**: The individual processor containing one or more Tensor Cores. Each core features specialized Matrix Multiply Units (MXUs) based on systolic array architectures.
- **TPU Pod**: A massive cluster of TPU chips connected via a high-speed, dedicated torus network.
- **TPU Slice**: A sub-division of a TPU Pod that can be allocated to a single user. For example, a `v6e-4` slice indicates 4 chips of the v6e (Trillium) generation.

### Accessing Cloud TPU in GCP

Accessing these resources is typically managed through Google Cloud Platform (GCP). For high-performance workloads requiring immediate availability, the `flex-start` provisioning model is often employed. A typical TPU creation command looks like:

```bash
gcloud alpha compute tpus queued-resources create tpu-node-name \
    --zone=southamerica-east1-c \
    --accelerator-type=v6e-4 \
    --runtime-version=v2-alpha-tpuv6e \
    --provisioning-model=flex-start
```

## Optimization Strategy

A naive JAX implementation of 3DGS, such as the one found in `train_fern_resume.py`, typically mirrors the standard PyTorch training loop: it dispatches a single optimization step from the Python host on every iteration, performs random data sampling in Python, and interleaves I/O with device computation. While functional, this approach fails to capitalize on JAX's core strengths—compilation and accelerator saturation—and often leads to significant dispatch overhead and idle TPU cores.

To achieve state-of-the-art performance, we must transition to a JAX-native architecture designed for TPU efficiency. The primary goal of `jax-gs` is to minimize host-accelerator communication and maximize the utilization of TPU systolic arrays (MXUs). We achieve this through five core optimization "Actions" that restructure the training and rendering pipeline into high-throughput, block-compiled execution units.

### Action I: JIT-Compiles Training Blocks

In JAX, every Python-to-accelerator dispatch incurs a non-trivial overhead. To mitigate this, we aggregate multiple training iterations into a single `jax.lax.scan` loop, which is then JIT-compiled into a single XLA graph.

```python
@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def train_block(state, rng_key, all_targets, all_w2cs,
                steps_per_block, camera_static, optimizer,
                fast_tpu_rasterizer, sh_degree):
    rng_key, subkey = jax.random.split(rng_key)
    # Sampling indices on-device (Action II)
    idxs = jax.random.randint(subkey, (steps_per_block,), 0, all_targets.shape[0])
    batch_targets = all_targets[idxs]
    batch_w2cs = all_w2cs[idxs]

    def one_step(carry, inputs):
        state = carry
        target, w2c = inputs
        state, loss, metrics = train_step(state, target, w2c, camera_static, optimizer, ...)
        return state, (loss, metrics)

    state, (losses, metrics) = jax.lax.scan(one_step, state, (batch_targets, batch_w2cs))
    return state, rng_key, losses, metrics
```

### Action II: On-Device Data & Sampling

To avoid the bottleneck of transferring images and camera matrices from the CPU host to the TPU device on every iteration, we store the entire dataset in on-device memory. Index sampling for mini-batches is performed using `jax.random` primitives directly on the TPU, ensuring that the training loop remains entirely self-contained within the accelerator.

```python
# Prepare data on device once
all_targets = jnp.stack(jax_targets)
all_w2cs = jnp.stack([c.W2C for c in jax_cameras])

# Inside the JIT-compiled train_block (Action I):
idxs = jax.random.randint(subkey, (steps_per_block,), 0, all_targets.shape[0])
batch_targets = all_targets[idxs]
batch_w2cs = all_w2cs[idxs]
```

### Action III: Asynchronous I/O

While the TPU handles the intensive computation, monitoring progress requires periodic rendering of validation views and saving `.ply` snapshots. We offload these tasks to a background thread pool on the host CPU, preventing I/O operations from blocking the primary training pipeline.

```python
# Background executor for I/O and rendering
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Inside training loop:
if curr_iter % 1000 == 0:
    snap_gaussians_dict = get_active_gaussians(curr_state)
    fut = executor.submit(
        save_artifacts_task, 
        snap_gaussians_dict, curr_iter, progress_dir, ply_path, 
        jax_cameras[0], fast_tpu_rasterizer, render, save_ply, sh_degree
    )
```

### Action IV: Hardware-Specific Rasterization (CPU/GPU vs. TPU)

The rasterization process involves projecting 3D Gaussians onto a 2D image plane and alpha-blending them. While a standard JAX rasterizer uses nested loops over tiles, our **Fast TPU Rasterizer** flattens the execution domain into `[num_tiles, 256]` arrays. This allows XLA to fuse operations into large matrix multiplications that saturate the MXUs. Furthermore, we use `jax.checkpoint` to trade re-computation for memory, avoiding High Bandwidth Memory (HBM) fragmentation.

```python
@jax.checkpoint
def scan_fn(carry, inputs):
    # ... alpha-blending logic ...
    power = -0.5 * (dx * dx * ic00 + 2.0 * dx * dy * ic01 + dy * dy * ic11)
    alpha = jnp.exp(power) * op
    # ...
    return (c_accum, T), None

# Vectorized execution across tiles and pixels simultaneously
(final_c, final_T), _ = jax.lax.scan(chunk_scan, (c_init, T_init), jnp.arange(MAX_TILE_CHUNKS))
```

### Action V: Multi-TPU Parallelism

For large-scale scenes or faster convergence, we utilize `jax.pmap` for data-parallel training across multiple TPU cores. Gradients and metrics are synchronized across the high-speed Torus network using `jax.lax.pmean`.

```python
@partial(jax.pmap, axis_name="batch", ...)
def train_block(state, ...):
    # ... computation ...
    def one_step(carry, inputs):
        # ... train_step_internal performs pmean on gradients ...
        return state, (loss, metrics)

    state, (losses, metrics) = jax.lax.scan(one_step, state, (batch_targets, batch_w2cs))
    avg_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)
    return state, losses, avg_metrics
```

Adaptive density control (cloning/splitting/pruning) is performed by unreplicating the state to a single "authoritative" device, executing the logic, and then re-broadcasting the result.

## Benchmark Results

Our benchmarks on the LLFF `room` dataset (504x378 resolution) demonstrate the massive performance gains achieved by our JAX-native optimizations. On a **Google Cloud TPU v6e-4 (Trillium)**, we observed the following:

### Rasterizer Optimization

The "Fast TPU Rasterizer" achieves a **~100x speedup** over the standard JAX implementation by maximizing MXU utilization and minimizing HBM latency.

| Metric | Standard Rasterizer | Fast TPU Rasterizer | Speedup |
| :--- | :--- | :--- | :--- |
| **Throughput (Steady State)** | ~0.09 it/s | ~9.6 it/s | **~107x** |
| **Convergence Time (3k steps)** | ~9.2 hours | ~5.2 minutes | **~107x faster** |

### Multi-Device Scaling

By utilizing `jax.pmap` and `jax.lax.scan`, we achieve near-linear scaling across multiple TPU cores. The efficiency remains high even as the complexity of the scene increases.

| Phase | Active Gaussians | Single Device Throughput | Multi-Device (4 TPUs) | Scaling Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **SH Degree 0** | ~17.5k | 9.6 img/s | 38.4 img/s | **4.0x** |
| **SH Degree 1** | ~34k | 6.6 img/s | 21.6 img/s | **3.3x** |

The "Fast TPU Rasterizer" achieves near-peak MXU saturation by replacing irregular memory access patterns with contiguous tensor operations, while `train_parallel.py` effectively hides communication overhead at scale.

## Conclusion

The transformation of 3D Gaussian Splatting into a static-shape, JIT-compatible architecture within JAX enables efficient training and rendering on modern accelerators. By prioritizing systolic array saturation, minimizing host-device communication, and leveraging the low-latency TPU interconnects, `jax-gs` provides a robust and scalable foundation for the next generation of radiance field research.
