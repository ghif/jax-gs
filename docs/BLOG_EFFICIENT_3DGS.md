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

### **Architectural Highlights**
Unlike general-purpose GPUs, TPUs are built around **Matrix Multiply Units (MXUs)**. These hardware blocks use a **systolic array** design to perform massive matrix multiplications with incredibly high throughput. 

| Version | Key Innovation | Topology | Memory |
| :--- | :--- | :--- | :--- |
| **v3** | Liquid Cooling | 2D Torus | 32 GB |
| **v4** | SparseCore | 3D Torus | 32 GB |
| **v5e** | Cost Efficiency | 2D Torus | 16 GB |
| **v5p** | Frontier Scale | 3D Torus | 95 GB |
| **v6e** | Trillium Arch | 2D Torus | 32 GB |

To achieve this performance, the TPU needs a constant, uninterrupted stream of data. JAX, combined with the **XLA (Accelerated Linear Algebra)** compiler, allows us to write high-level Python code that is compiled into a single, optimized "graph" for the TPU. If our code frequently jumps back to Python or waits for the CPU, the TPU's MXUs sit idle—a state known as being "host-bound." Our optimizations focus on breaking this host-bound barrier.

---

## 2. The Bottleneck: The Naive Approach

Our initial implementation, `train_fern_resume.py`, followed a traditional deep learning training loop structure familiar to many PyTorch users:

```python
# From train_fern_resume.py
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
1.  **Dispatch Overhead**: Python has to tell the TPU what to do for every single step. The time it takes for Python to "talk" to the TPU can be longer than the actual training step itself.
2.  **Host-Device Communication**: On every iteration, we pick a random image on the CPU and send it to the TPU. This creates a constant traffic jam on the PCIe bus.
3.  **Synchronous I/O**: Every 100 steps, the entire training process stops. The TPU waits for the CPU to render an image, pull it into RAM, and write it to a slow disk.

---

## 3. Optimization 1: JIT-Compiled Training Blocks

The first major breakthrough in `train.py` was moving the loop itself onto the TPU using `jax.lax.scan`. Instead of running one step at a time, we compiled a "block" of 500 steps.

```python
# From train.py
@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def train_block(state, rng_key, all_targets, all_w2cs, steps_per_block, ...):
    def one_step(carry, _):
        # ... perform one training step ...
        return (state, key), loss

    # The entire loop runs on the TPU
    (state, rng_key), losses = jax.lax.scan(one_step, (state, rng_key), None, length=steps_per_block)
    return state, rng_key, losses
```

By using `jax.lax.scan`, XLA compiles the entire 500-step loop into a single TPU program. Python dispatches the work once, and the TPU runs for 500 iterations without ever looking back at the host. This effectively deletes dispatch overhead from the performance equation.

---

## 4. Optimization 2: On-Device Data & Sampling

To solve the communication bottleneck, `train.py` loads all training images and camera matrices onto the TPU memory upfront.

```python
# Prepare data on device
all_targets = jnp.stack(jax_targets)
all_w2cs = jnp.stack([c.W2C for c in jax_cameras])

# Inside the JIT-compiled block:
idx = jax.random.randint(subkey, (), 0, all_targets.shape[0])
target = all_targets[idx] # Zero-copy selection on TPU
```

We replaced Python's `random.randint` with `jax.random.randint`. Because the indices are generated on the TPU and the images are already in TPU memory, there is **zero data transfer** during the 500-step training block.

---

## 5. Optimization 3: Asynchronous I/O

Finally, we tackled the I/O bottleneck. In `train.py`, we don't let the TPU wait for the filesystem. We use a background thread to handle all logging and checkpointing.

```python
# Background executor for I/O and rendering
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

for b in pbar:
    curr_state, losses = train_block(...)
    
    if curr_iter % 1000 == 0:
        # Snap the state and offload to a background thread
        snap_gaussians = curr_state[0]
        executor.submit(save_artifacts_task, snap_gaussians, ...)
```

The TPU finishes a block and immediately starts the next one. Meanwhile, a CPU thread takes a "snapshot" of the Gaussians, renders the progress image, and writes the `.ply` file in the background. The training loop never pauses.

---

## 6. Scaling Up: Multi-TPU Parallelism

For even larger scenes or faster convergence, `train_parallel.py` introduces Data Parallelism across multiple TPU cores.

```python
# From train_parallel.py
@partial(jax.pmap, axis_name='batch', ...)
def train_block(state, rng_key, all_targets, all_w2cs, ...):
    # This function now runs on ALL TPU cores simultaneously
    # Each core processes its own random camera
```

Using `jax.pmap` (Parallel Map), we replicate the Gaussian model across every available TPU device. During each step of the `scan`, every TPU core picks a *different* random camera, computes its own gradient, and then performs a collective "all-reduce" to synchronize the weights. This allows us to scale our effective batch size linearly with the number of TPU devices.

---

## 7. Conclusion

By shifting our mindset from "Python calling kernels" to "Compiling entire loops for the device," we achieved orders of magnitude speed improvements. The core philosophy of efficient JAX development remains:
1.  **Minimize Dispatch**: Use `jax.lax.scan` to run loops on-device.
2.  **Keep Data Local**: Stack your datasets and sample on-device.
3.  **Don't Wait on I/O**: Use asynchronous background threads for saving and logging.

With these optimizations, 3D Gaussian Splatting on TPUs becomes not just viable, but incredibly performant.
