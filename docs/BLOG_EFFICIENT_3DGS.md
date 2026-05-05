# Efficient 3D Gaussian Splatting in JAX: From Naive Loops to TPU Saturation

3D Gaussian Splatting (3DGS) has revolutionized real-time radiance field rendering. However, training these models efficiently requires more than just porting CUDA kernels to Python. In this article, we explore how we optimized a 3DGS training pipeline in JAX, moving from a naive implementation that left hardware idling to a highly optimized system that fully saturates Google's Tensor Processing Units (TPUs).

---

## 1. Understanding Google's TPU Technology

To understand our optimizations, we must first understand the target hardware. Google's **Tensor Processing Units (TPUs)** are application-specific integrated circuits (ASICs) designed from the ground up to accelerate machine learning workloads.

### **A Brief History of TPUs**
Since their secret debut in 2015, TPUs have evolved from specialized inference engines into the backbone of global AI:
*   **TPU v1 (2015):** A pure inference chip that powered Google Search and AlphaGo.
*   **TPU v2 (2017):** The first version capable of training, introducing the **bfloat16** format.
*   **TPU v3 (2018):** Doubled performance and introduced **liquid cooling** to handle the heat of massive scale.
*   **TPU v4 (2021):** Introduced **3D Torus** network topology and **SparseCore** for embedding acceleration.
*   **TPU v5 (2023):** Split into **v5e** (Efficient/Cost-optimized) and **v5p** (Performance flagship for models like Gemini).
*   **Trillium / TPU v6e (2024):** The latest generation, featuring a 256x256 MXU and 4.7x the peak compute of v5e.
*   **Ironwood / TPU7x (2025):** Employs a **dual-chiplet** architecture and 192GB of HBM3e, optimized for massive-scale inference and frontier "reasoning" models.

### **Architectural Highlights**
Unlike general-purpose GPUs, TPUs are built around **Matrix Multiply Units (MXUs)**. These hardware blocks use a **systolic array** design to perform massive matrix multiplications with incredibly high throughput. 

| Version | Key Innovation | Topology | Memory |
| :--- | :--- | :--- | :--- |
| **v3** | Liquid Cooling | 2D Torus | 32 GB |
| **v4** | SparseCore | 3D Torus | 32 GB |
| **v5e** | Cost Efficiency | 2D Torus | 16 GB |
| **v5p** | Frontier Scale | 3D Torus | 95 GB |
| **v6e** | Trillium Arch | 2D Torus | 32 GB |
| **Ironwood** | Dual-Chiplet | 3D Torus | 192 GB |

To achieve this performance, the TPU needs a constant, uninterrupted stream of data. JAX, combined with the **XLA (Accelerated Linear Algebra)** compiler, allows us to write high-level Python code that is compiled into a single, optimized "graph" for the TPU. If our code frequently jumps back to Python or waits for the CPU, the TPU's MXUs sit idle—a state known as being "host-bound." Our optimizations focus on breaking this host-bound barrier.

---

## 2. The Bottleneck: The Naive Approach

Our initial implementation followed a traditional deep learning training loop structure familiar to many PyTorch users:

```python
# The Naive Loop
for i in pbar:
    idx = random.randint(0, len(jax_cameras)-1)
    cam = jax_cameras[idx]
    target = jax_targets[idx]
    
    # Single step dispatched to device
    state, loss, metrics = train_step(state, target, cam.W2C, camera_static, optimizer)
    
    if i % 100 == 0:
        img = render(state[0], jax_cameras[0]) # Synchronous render
        save_ply(...) # Synchronous I/O
```

This approach has three fatal performance flaws:
1.  **Dispatch Overhead**: Python has to tell the TPU what to do for every single step.
2.  **Host-Device Communication**: Picking random images on the CPU and sending them to the TPU creates a bottleneck.
3.  **Synchronous I/O**: The entire training process stops to wait for disk writes.

---

## 3. Optimization 1: JIT-Compiled Training Blocks

The first breakthrough was moving the loop itself onto the TPU using `jax.lax.scan`. Instead of running one step at a time, we compiled a "block" of 100 steps.

```python
# From train.py
@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def train_block(state, rng_key, all_targets, all_w2cs, steps_per_block, camera_static, optimizer, fast_tpu_rasterizer, sh_degree):
    def one_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        
        # Sample camera index
        idx = jax.random.randint(subkey, (), 0, all_targets.shape[0])
        target = all_targets[idx]
        w2c = all_w2cs[idx]
        
        # Perform training step
        state, loss, metrics = train_step(state, target, w2c, camera_static, optimizer, fast_tpu_rasterizer=fast_tpu_rasterizer, sh_degree=sh_degree)
        
        return (state, key), loss

    # The entire loop runs on the TPU as a single XLA program
    (state, rng_key), losses = jax.lax.scan(one_step, (state, rng_key), None, length=steps_per_block)
    return state, rng_key, losses
```

This effectively deletes dispatch overhead from the performance equation. The TPU runs for 100 iterations without ever looking back at the host.

---

## 4. Optimization 2: On-Device Data & Sampling

To solve the communication bottleneck, we load all training images and camera matrices onto the TPU memory upfront. Indices are generated on the TPU using `jax.random.randint`, resulting in **zero data transfer** during training blocks.

```python
# From train.py
# 1. Prepare data on device (all images and matrices loaded once)
all_targets = jnp.stack(jax_targets)
all_w2cs = jnp.stack([c.W2C for c in jax_cameras])

# 2. Inside the JIT-compiled train_block:
# Sampling happens entirely on-device
idx = jax.random.randint(subkey, (), 0, all_targets.shape[0])
target = all_targets[idx]
w2c = all_w2cs[idx]
```

---

## 5. Optimization 3: Asynchronous I/O

We use a background `ThreadPoolExecutor` to handle logging and checkpointing. While the TPU is busy with the next training block, the CPU thread takes a "snapshot" of the Gaussians, renders a progress image, and writes the `.ply` file. The training loop never pauses.

```python
# From train.py
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

for b in pbar:
    # TPU is busy executing the next block...
    curr_state, curr_rng, losses = train_block(...)

    # Dispatch I/O to background thread without blocking the TPU
    def save_async(state_snapshot, path):
        save_ply(state_snapshot.gaussians, path)
        
    executor.submit(save_async, jax.device_get(curr_state), ply_path)
```

---

## 6. Optimization 4: JIT-Compatible Adaptive Density Control

3DGS requires dynamically cloning, splitting, and pruning Gaussians. However, JIT-compiled functions require **static array shapes**. Our solution is the **Padded Buffer Strategy**:

1.  **Fixed-Size State**: We initialize a `DensityState` with a fixed buffer (e.g., 200,000 slots).
2.  **Active Mask**: A boolean mask tracks which slots contain valid Gaussians.
3.  **Vectorized Routing**: Instead of dynamic slicing, we use `jnp.argsort` on masks to "pack" valid indices and scatter new Gaussians into empty slots—all within a statically-shaped XLA graph.

```python
# From jax_gs/training/density.py
# --- Vectorized Routing ---
# Compact indices for sources and destinations using argsort
empty_indices = jnp.argsort((~will_be_empty).astype(jnp.int8))
clone_src_indices = jnp.argsort((~clone_mask).astype(jnp.int8))
split_src_indices = jnp.argsort((~split_mask).astype(jnp.int8))

# Identify which source to read for each empty slot
src_to_read = jnp.where(is_clone_dest, clone_src_indices[idx], 0)
src_to_read = jnp.where(is_split_dest, split_src_indices[idx - num_clones], src_to_read)

# Execute Data Movement with static shapes
gather_indices = jnp.arange(MAX_GAUSSIANS)
gather_indices = gather_indices.at[empty_indices].set(new_values)
new_means = g.means[gather_indices]
```

4.  **Optimizer Synchronization**: Crucially, we use `jax.tree_util.tree_map` to reorder the **optimizer momentum buffers** (mu, nu) whenever Gaussians are moved. This keeps the optimization history in perfect sync with the geometry.

---

## 7. Optimization 5: Sophisticated Radiance Field Tuning

To handle complex indoor scenes like the "room" dataset, we implemented advanced radiance field optimization techniques:

*   **Multi-Parameter Optimization**: Using `optax.multi_transform`, we apply different learning rates to different parameters. Opacities get a high LR (`0.05`) for fast convergence, while positions (`means`) use a scene-scaled exponential decay.

```python
# From train.py
optimizer = optax.multi_transform(
    {
        "means": optax.adam(learning_rate=means_lr_schedule),
        "scales": optax.adam(learning_rate=0.005),
        "quaternions": optax.adam(learning_rate=0.001),
        "opacities": optax.adam(learning_rate=0.05),
        "sh_coeffs": optax.adam(learning_rate=0.0025),
    },
    param_labels
)
```

*   **Gradient Rescaling**: JAX typically averages loss over pixels (`jnp.mean`). Since densification thresholds expect a sum-of-errors, we scale our accumulated positional gradients by the **total pixel count** ($H \times W$).
*   **SH Degree Scheduling**: We start training at Degree 0 (diffuse only) and gradually increase view-dependency every 1,000 steps. This "geometric warmup" prevents early optimization instability.

```python
# From train.py
for b in pbar:
    # SH Degree Scheduling: Increase degree every 1000 iterations
    curr_iter = b * steps_per_block
    sh_degree = min(3, curr_iter // 1000)
    curr_state, _, _ = train_block(..., sh_degree=sh_degree)
```

*   **Stable Radii Accumulation**: Gaussians are pruned if their 2D screen radius exceeds a threshold. However, computing the maximum radius across parallel devices (`jax.lax.pmax`) can be corrupted if a Gaussian passes too close to the camera plane ($z \approx 0$), causing an artificially massive radius. We strictly mask these invalid projections before updating the `max_radii` state to prevent erroneous, aggressive pruning.
*   **Relaxed Screen Size Threshold**: The default threshold of 20 pixels used in many baseline implementations is too restrictive for modern high-resolution scenes. We increased the `max_screen_size` pruning threshold to `512` pixels, allowing geometry to naturally expand to capture large continuous regions (like walls or backgrounds).
*   **Masked Rendering**: Our renderer strictly ignores inactive buffer slots during projection and sorting, dramatically reducing TPU memory pressure and noise.

---

## 8. Optimization 6: Hardware-Specific Rasterization (CPU/GPU vs. TPU)

A critical realization during development was that the optimal JAX code for a GPU is not always the optimal code for a TPU. To maximize performance across platforms, we implemented two distinct rasterizers: `rasterizer.py` (standard) and `rasterizer_tpu.py` (TPU-optimized).

The core differences lie in how they structure memory access and vectorization for the XLA compiler:

1.  **Vectorization Strategy (Nested vs. Flat)**:
    *   **Standard (`rasterizer.py`)**: Uses a nested approach. It parallelizes over tiles using `jax.vmap`, and inside that, parallelizes over the 256 pixels within the tile using another `jax.vmap`. This is natural and works well on GPUs.

```python
# From jax_gs/renderer/rasterizer.py
def rasterize_single_tile(tile_idx):
    # ... logic for one tile ...
    def blend_pixel(p_coord, p_valid):
        # ... blending logic for one pixel ...
        return final_color

    # Parallelize over pixels in the tile
    tile_image = jax.vmap(blend_pixel)(pixel_coords, pixel_valid)
    return tile_image.reshape(tile_size, tile_size, 3)

# Parallelize over all tiles in the image
all_tiles = jax.vmap(rasterize_single_tile)(jnp.arange(num_tiles))
```

    *   **TPU (`rasterizer_tpu.py`)**: TPUs prefer massive, continuous matrix operations to saturate their Matrix Multiply Units (MXUs). The TPU rasterizer flattens the tile and pixel dimensions into a single massive axis (`[num_tiles, 256]`). The core blending loop (`scan_fn`) processes a single Gaussian across *all pixels in all tiles simultaneously*.

```python
# From jax_gs/renderer/rasterizer_tpu.py
# Pre-calculate global pixel coordinates for EVERY pixel in the image, 
# grouped by tile. Shape: [num_tiles, 256]
px = (tx[:, None] * TILE_SIZE).astype(jnp.float32) + (idx % TILE_SIZE)[None, :].astype(jnp.float32) + 0.5
py = (ty[:, None] * TILE_SIZE).astype(jnp.float32) + (idx // TILE_SIZE)[None, :].astype(jnp.float32) + 0.5

@jax.checkpoint
def scan_fn(carry, i):
    # Vectorized across [num_tiles, 256] flat dimension
    dx = px - mu_x 
    dy = py - mu_y
    # ... blending logic ...
```

2.  **Memory Access Patterns (Random vs. Broadcasted Gather)**:
    *   **Standard**: Inside the inner loop, it dynamically slices (`jnp.take`) the specific Gaussians that overlap the current tile.
    *   **TPU**: Dynamic slicing inside a fast loop is terrible for TPU performance. Instead, we use a **Broadcasted Gather**. Before the loop starts, we prefetch *all* Gaussian parameters for *every* tile into massive tensors (e.g., `[num_tiles, BLOCK_SIZE, 3]` for colors).

```python
# From jax_gs/renderer/rasterizer_tpu.py
# BROADCASTED GATHER: Construct indices for all Gaussians across all tiles.
# Resulting shape: [num_tiles, BLOCK_SIZE]
all_tile_indices = tile_starts[:, None] + jnp.arange(BLOCK_SIZE)[None, :]

# Prefetch Gaussian data for all tiles at once
tile_gids = valid_ids[all_tile_indices] 
g_means = means2D[tile_gids]            # [num_tiles, BLOCK_SIZE, 2]
g_cols = colors[tile_gids]              # [num_tiles, BLOCK_SIZE, 3]

@jax.checkpoint
def scan_fn(carry, i):
    # Processes the i-th Gaussian for ALL pixels in ALL tiles simultaneously
    # No random access lookups here!
    mu_x = g_means[:, i, 0][:, None] # [num_tiles, 1]
    dx = px - mu_x                   # [num_tiles, 256]
    # ... blending logic ...
```

3.  **Memory Efficiency (`jax.checkpoint`)**:
    *   Because the TPU rasterizer trades memory for speed via the Broadcasted Gather, it risks Out-Of-Memory (OOM) errors during the backward pass (autodiff). We solve this by heavily applying `@jax.checkpoint` to the inner `scan` loop. This forces JAX to discard intermediate activations during the forward pass and recompute them on-the-fly during backpropagation, keeping memory scaling linear rather than exploding.

---

## 9. Scaling Up: Multi-TPU Parallelism

For larger scenes, we use `jax.pmap` to replicate the model across multiple TPU cores. Each core picks a different random camera, computes local gradients, and performs a collective **all-reduce** to synchronize. This scales our effective batch size linearly with the hardware.

```python
# From train_parallel.py
@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(4, 5, 6, 7, 8))
def train_block(state, rng_key, all_targets, all_w2cs, ...):
    # Each device runs its own scan loop
    (state, rng_key), losses = jax.lax.scan(one_step, (state, rng_key), ...)
    return state, rng_key, losses

# From jax_gs/training/trainer.py
# Gradients and metrics are averaged across devices
grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), grads)
radii_max = jax.lax.pmax(radii_masked, axis_name='batch')
```

---

## 10. Conclusion

By shifting from "Python calling kernels" to "Compiling entire pipelines," we achieved orders of magnitude speed improvements. Gaussian Splatting on TPUs is no longer just a port—it's a highly specialized, hardware-saturated engine.

**Core Philosophy:**
1.  **Minimize Dispatch**: Use `jax.lax.scan` for loops.
2.  **Stay Static**: Use padded buffers and masks for dynamic data.
3.  **Sync the History**: Keep optimizer moments aligned with parameter movement.
4.  **Scale the Gradients**: Match densification thresholds to your loss normalization.

With these optimizations, 3DGS training on TPUs is not just incredibly fast—it's mathematically robust and stable on the most complex real-world scenes.