# Optimizing 3D Gaussian Splatting Training on Multi-TPU setups

This document outlines strategies for accelerating the 3D Gaussian Splatting (3DGS) training process in `train_fern.py` by exploiting parallelism across multiple TPU cores.

Currently, `train_fern.py` relies on standard JIT compilation (`jax.jit`), which implicitly runs on a single device unless specifically told otherwise. To fully utilize a multi-TPU machine (e.g., a TPU v3-8 or v4-8), we must explicitly parallelize the computation.

## 1. Data Parallelism (The Primary Strategy)

The most straightforward and effective approach for 3DGS training on TPUs is **Data Parallelism**. In this setup, the Gaussian parameters (the model) are replicated across all TPU cores, but each core processes a *different* camera view (image) in a single training step.

### How it Works:
1. **Replicate Model:** The `gaussians` state and optimizer state are copied to all available TPU devices.
2. **Distribute Data:** Instead of picking 1 random camera per step, pick $N$ random cameras (where $N$ is the number of TPU devices, or a multiple thereof). Each device receives a different target image and camera pose.
3. **Parallel Forward/Backward Pass:** Each device independently renders its assigned view and computes the gradients for the loss.
4. **Gradient Aggregation (All-Reduce):** The gradients computed by each device are averaged together across all devices using an operation like `jax.lax.pmean`.
5. **Update Model:** Every device applies the identical, averaged gradients to its local copy of the model, ensuring all devices stay synchronized.

### Implementation in JAX:

JAX provides two main APIs for this: `jax.pmap` (older, explicit SPMD) and `jax.sharding` with `jax.jit` (newer, compiler-driven).

**A. Using `jax.pmap` (Similar to `train_tpu.py`):**

```python
import jax
from functools import partial

# 1. Replicate state
num_devices = jax.device_count()
# Using jax.device_put or jax.tree_map with sharding to replicate
replicated_state = jax.device_put_replicated(state, jax.devices())

# 2. Define parallel step
@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,))
def p_train_step(state, target_images, w2cs, camera_static, optimizer):
    # state: replicated on each device
    # target_images, w2cs: sharded across devices (batch size = num_devices)
    
    def loss_fn(params):
        # Render and compute loss for the local camera view
        ... 
        return loss
        
    loss, grads = jax.value_and_grad(loss_fn)(state[0])
    
    # Crucial step: Average gradients across all devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Update local state
    updates, new_opt_state = optimizer.update(grads, state[1], state[0])
    new_params = optax.apply_updates(state[0], updates)
    
    return (new_params, new_opt_state), loss

# 3. Training Loop
for i in range(num_iterations // num_devices):
    # Sample a batch of cameras
    batch_targets = ... # shape: (num_devices, H, W, 3)
    batch_w2cs = ...    # shape: (num_devices, 4, 4)
    
    replicated_state, loss = p_train_step(
        replicated_state, batch_targets, batch_w2cs, camera_static, optimizer
    )
```

**B. Using `jax.sharding` and `jax.jit` (Modern Approach):**

Instead of `pmap`, you define a `jax.sharding.Mesh` and `PartitionSpec` to tell the compiler how data should be laid out. `jax.jit` automatically infers the necessary communication (like all-reduce).

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

mesh = Mesh(jax.devices(), ('batch',))

# Replicate model state
model_sharding = NamedSharding(mesh, P()) # Empty P() means replicate
state = jax.device_put(state, model_sharding)

# Shard data across the 'batch' axis
data_sharding = NamedSharding(mesh, P('batch'))

@jax.jit
def train_step(state, batch_targets, batch_w2cs):
    # Standard vmap over the batch to compute per-example loss/grads
    def per_example_loss(params, target, w2c):
        ...
        return loss
        
    # vmap the loss function across the batch dimension
    batch_loss_fn = jax.vmap(per_example_loss, in_axes=(None, 0, 0))
    
    def mean_loss(params):
        return jnp.mean(batch_loss_fn(params, batch_targets, batch_w2cs))
        
    loss, grads = jax.value_and_grad(mean_loss)(state[0])
    
    # Optimizer update
    ...
    return new_state, loss

# In the loop, ensure data is sharded before passing to train_step
batch_targets = jax.device_put(batch_targets, data_sharding)
batch_w2cs = jax.device_put(batch_w2cs, data_sharding)
```

## 2. Gradient Accumulation (To increase effective batch size)

If memory permits, or if you want a larger effective batch size than the number of TPUs, you can combine Data Parallelism with Gradient Accumulation.

1. Compute gradients on $N$ devices.
2. Store the gradients (don't update the model yet).
3. Repeat for $M$ micro-batches.
4. Sum/average all accumulated gradients.
5. Perform an all-reduce across devices.
6. Update the model.

This stabilizes the optimization, which can be critical for 3DGS as the number of points grows.

## 3. Pipeline Parallelism / Model Sharding (Advanced)

*Applicability for 3DGS: Generally low, unless the point cloud is exceptionally massive.*

If the number of Gaussians grows so large that the model parameters ($M$) and optimizer states (Momentum, Variance) can no longer fit in the HBM of a single TPU core (typically 16GB or 32GB), Data Parallelism fails (because it requires replicating the model).

In this scenario, you would need **Model/Tensor Parallelism** (sharding the `gaussians` arrays across devices). However, the projection and rasterization steps in 3DGS are inherently difficult to shard across points without heavy cross-device communication (all-to-all), which TPUs can struggle with if not perfectly optimized.

**Recommendation:** Stick to Data Parallelism for 3DGS unless you hit hard Out-Of-Memory (OOM) errors during initialization.

## Summary of Actionable Steps for `train_fern.py`:

1.  **Modify the training loop** to sample a batch of cameras equal to `jax.device_count()`.
2.  **Update `train_step`** to handle batched inputs. If it's currently written for a single camera, you will need to wrap the forward/backward pass in a `jax.vmap` before JIT compiling it, or use `jax.pmap`.
3.  **Ensure Sharding:** Use `jax.device_put_replicated` (or `NamedSharding`) for the `gaussians` and `opt_state`, and distribute the batch of images across the devices.
4.  **Reduce Iterations:** If you process 8 images per step instead of 1, you can reduce `num_iterations` by roughly a factor of 8 to achieve a similar amount of training (though convergence dynamics may change slightly with larger batch sizes).