# Efficient 3D Gaussian Splatting in JAX-GS

## A Static-Shape Reinterpretation of Kerbl et al. (2023)

### Abstract

This note documents the current `jax-gs` implementation of 3D Gaussian Splatting (3DGS) after the recent convergence and systems updates. The core objective remains the same as Kerbl et al. (2023): optimize anisotropic 3D Gaussians with interleaved densification, pruning, and spherical harmonics (SH) color fitting. The main departure is implementation strategy. Instead of a PyTorch training loop coupled to custom CUDA rasterization, `jax-gs` reformulates the pipeline as pure JAX/XLA programs with static-shape state, block-compiled optimization, and hardware-specific rasterization paths for standard accelerators and TPUs. Recent code changes focus less on raw throughput claims and more on maintaining numerical and structural stability under JIT compilation: masked rendering of inactive padded slots, optimizer-state reordering during densification, gradient rescaling for density heuristics, interaction-pressure metrics from the renderer, and health-gated SH promotion and densification.

---

## 1. Motivation

Kerbl et al. showed that explicit 3D Gaussians can replace neural volume integration while preserving high visual quality and real-time rendering. However, the original implementation assumes a systems stack that is natural for CUDA: variable-length point sets, custom rasterization kernels, and dynamic memory movement during clone/split/prune operations. Those assumptions do not transfer directly to JAX, where high performance depends on static array shapes and compiling large regions of the program into a single XLA graph.

The current `jax-gs` codebase therefore solves a different engineering problem:

1. Preserve the optimization logic of 3DGS.
2. Express the full renderer and training loop in pure JAX.
3. Keep the state JIT-compatible despite adaptive density control.
4. Add explicit quality guards so densification does not destabilize the tiled rasterizer.

---

## 2. System Overview

At a high level, the implementation is organized around four stages:

1. A padded Gaussian state in `jax_gs/training/density.py`.
2. Projection, tile assignment, sorting, and rasterization in `jax_gs/renderer/`.
3. A block-compiled training loop in `train.py` or `train_parallel.py`.
4. Asynchronous host-side artifact export for progress images and `.ply` snapshots.

The most important systems decision is that Gaussian count is treated as a fixed-capacity buffer rather than a dynamically resized list. This single choice drives most of the subsequent design.

---

## 3. Core Training Formulation

### 3.1 Block-Compiled Optimization

The single-device trainer no longer dispatches one optimizer step at a time from Python. Instead, it samples a block of camera indices on device and executes the block with `jax.lax.scan`.

```python
@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def train_block(state, rng_key, all_targets, all_w2cs,
                steps_per_block, camera_static, optimizer,
                fast_tpu_rasterizer, sh_degree):
    rng_key, subkey = jax.random.split(rng_key)
    idxs = jax.random.randint(subkey, (steps_per_block,), 0, all_targets.shape[0])
    batch_targets = all_targets[idxs]
    batch_w2cs = all_w2cs[idxs]

    def one_step(carry, inputs):
        state = carry
        target, w2c = inputs
        state, loss, metrics = train_step(
            state, target, w2c, camera_static, optimizer,
            fast_tpu_rasterizer=fast_tpu_rasterizer,
            sh_degree=sh_degree,
        )
        return state, (loss, metrics)

    state, (losses, metrics) = jax.lax.scan(one_step, state, (batch_targets, batch_w2cs))
    return state, rng_key, losses, metrics
```

This is a direct response to the JAX execution model. In Kerbl et al., per-step Python control is not itself the bottleneck because the heavy work happens in CUDA kernels. In `jax-gs`, leaving the loop on the host would reintroduce dispatch overhead and fragment compilation.

### 3.2 Parameter-Specific Optimization

The optimizer follows the 3DGS idea of parameter-group-specific learning rates, but the current implementation uses an exponential decay schedule for positions and tuned constants for the remaining parameter groups.

```python
means_lr_init = 0.00016 * extent * 3.0
means_lr_end = 0.0000016 * extent * 3.0
means_lr_schedule = optax.exponential_decay(
    init_value=means_lr_init,
    transition_steps=num_iterations,
    decay_rate=means_lr_end / means_lr_init,
)

optimizer = optax.multi_transform(
    {
        "means": optax.adam(learning_rate=means_lr_schedule),
        "scales": optax.adam(learning_rate=0.005),
        "quaternions": optax.adam(learning_rate=0.001),
        "opacities": optax.adam(learning_rate=0.05),
        "sh_coeffs": optax.adam(learning_rate=0.0025),
    },
    param_labels,
)
```

This is one of the recent stability changes: the code moved away from more aggressive scaling and back toward a schedule that decays smoothly over the full run.

---

## 4. Static-Shape Adaptive Density Control

### 4.1 Fixed-Capacity State

The major algorithmic adaptation relative to the original 3DGS implementation is the introduction of `DensityState`, which holds:

- padded Gaussian parameters,
- optimizer state,
- an `active_mask`,
- per-Gaussian gradient accumulators,
- visibility counts,
- maximum observed 2D radii.

```python
@chex.dataclass
class DensityState:
    gaussians: Gaussians
    opt_state: optax.OptState
    active_mask: jnp.ndarray
    num_active: jnp.ndarray
    grad_accum: jnp.ndarray
    denom: jnp.ndarray
    max_radii: jnp.ndarray
```

Kerbl et al. manipulate a logically dynamic set of Gaussians. `jax-gs` instead allocates a static buffer once and uses `active_mask` to distinguish valid from inactive slots. This is the key difference that makes densification JIT-compatible.

### 4.2 Clone/Split/Prune Without Dynamic Resizing

Recent versions of the code perform densification through vectorized routing over padded arrays. Empty destinations and source indices are compacted with `argsort`, then a static `gather_indices` permutation is used to materialize the next state.

```python
empty_indices = jnp.argsort((~will_be_empty).astype(jnp.int8))
clone_src_indices = jnp.argsort((~clone_mask).astype(jnp.int8))
split_src_indices = jnp.argsort((~split_mask).astype(jnp.int8))

idx = jnp.arange(MAX_GAUSSIANS)
is_clone_dest = idx < num_clones
is_split_dest = (idx >= num_clones) & (idx < num_clones + num_splits)

src_to_read = jnp.where(is_clone_dest, clone_src_indices[idx], 0)
src_to_read = jnp.where(is_split_dest, split_src_indices[idx - num_clones], src_to_read)

gather_indices = jnp.arange(MAX_GAUSSIANS)
gather_indices = gather_indices.at[empty_indices].set(
    jnp.where(is_clone_dest | is_split_dest, src_to_read, empty_indices)
)
```

The point here is not merely functional equivalence. The routing must preserve static shapes so that the same XLA program can execute before and after densification.

### 4.3 Optimizer-State Reordering

One subtle but important recent fix is that densification now reorders the optimizer state along with the Gaussian parameters.

```python
def reorder_opt_state(opt_state, gather_indices):
    def reorder_leaf(x):
        if hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == gather_indices.shape[0]:
            return x[gather_indices]
        return x
    return jax.tree_util.tree_map(reorder_leaf, opt_state)
```

Without this step, Adam moments become attached to the wrong Gaussian after clone/split/prune permutations. That error is easy to miss because the code still runs, but convergence degrades.

### 4.4 Density Heuristics After the Recent Fixes

The recent trainer changes corrected two issues in the density statistics:

1. Positional gradients are rescaled by `W * H * 3` to compensate for loss normalization under `jnp.mean`.
2. The visibility denominator is updated only for projected active Gaussians.

```python
grad_2d_mag = jnp.linalg.norm(grads.means, axis=-1) * (z / focal) * (float(W) * float(H) * 3.0)

visible = extras["valid_mask"] & active
next_denom = state.denom + jnp.where(visible, 1, 0)
```

This is a JAX-specific correction, not a conceptual change to 3DGS. Kerbl et al. rely on density heuristics calibrated to their own loss and renderer. Once the implementation uses pixel-mean losses and a different compiled execution structure, the heuristic scale must be restored explicitly.

---

## 5. Renderer Design

### 5.1 Tile Assignment and Bit-Packed Sort

Like the original 3DGS pipeline, `jax-gs` uses tile-based culling and front-to-back compositing. The implementation computes Gaussian-to-tile interactions and sorts them with a packed integer key:

```python
DEPTH_BITS = 13
sort_tile_ids = jnp.where(valid_interactions, flat_tile_ids, num_tiles_total)
depth_i32_full = jax.lax.bitcast_convert_type(flat_depths, jnp.int32)
depth_quant = depth_i32_full >> (31 - DEPTH_BITS)
key = (sort_tile_ids << DEPTH_BITS) | depth_quant
sorted_keys, sorted_gaussian_ids = jax.lax.sort_key_val(key, flat_gaussian_ids)
```

The conceptual role matches Kerbl et al.: group by tile, order by depth, then blend front to back. The difference is that the operation is expressed as pure array primitives rather than a custom CUDA rasterizer.

### 5.2 Standard Rasterizer Path

The standard JAX rasterizer uses nested structure: tile parallelism outside, pixel-level blending inside, and chunked front-to-back accumulation with `lax.scan`.

```python
def blend_pixel(p_color, p_T, p_coord, p_valid):
    def scan_fn(inner_carry, i):
        accum_c, trans = inner_carry
        is_active = local_mask[i] & (trans > 1e-4)
        mu = t_means[i]
        dx = p_coord[0] - mu[0]
        dy = p_coord[1] - mu[1]
        power = -0.5 * (dx * dx * t_ic00[i] + dx * dy * t_ic01_2[i] + dy * dy * t_ic11[i])
        alpha = jnp.exp(power) * t_op_vec[i]
        alpha = jnp.where((power > -10.0) & is_active, jnp.minimum(0.99, alpha), 0.0)
        new_T = trans * (1.0 - alpha)
        new_color = accum_c + (alpha * trans) * t_cols[i]
        return (new_color, new_T), None

    return jax.lax.scan(scan_fn, (p_color, p_T), jnp.arange(BLOCK_SIZE))[0]
```

Two current constants matter here:

- `BLOCK_SIZE = 512`
- `MAX_TILE_CHUNKS = 8`

This yields a hard interaction budget per tile of `4096`. The code now exposes overflow and truncation statistics so training can react when the renderer is under excessive pressure.

### 5.3 TPU Rasterizer Path

For TPUs, the implementation switches to a different execution strategy in `rasterizer_tpu.py`. Tiles and pixels are flattened into `[num_tiles, 256]`, and Gaussian data are gathered in chunks for all tiles simultaneously.

```python
px = (tx[:, None] * TILE_SIZE).astype(jnp.float32) + (idx % TILE_SIZE)[None, :].astype(jnp.float32) + 0.5
py = (ty[:, None] * TILE_SIZE).astype(jnp.float32) + (idx // TILE_SIZE)[None, :].astype(jnp.float32) + 0.5

@jax.checkpoint
def scan_fn(carry, inputs):
    c_accum, T = carry
    mu_x, mu_y, ic00, ic01, ic11, op, cols, is_active_local = inputs
    dx = px - mu_x[:, None]
    dy = py - mu_y[:, None]
    power = -0.5 * (dx * dx * ic00[:, None] + 2.0 * dx * dy * ic01[:, None] + dy * dy * ic11[:, None])
    alpha = jnp.exp(power) * op[:, None]
    ...
```

This is not simply a faster version of the same code. It is a different memory-access pattern designed for XLA on TPUs:

- larger contiguous tensor operations,
- fewer random gathers inside the innermost loop,
- rematerialization via `jax.checkpoint` to control backward memory usage.

### 5.4 View-Dependent Color and Masked Rendering

An older simplification in the renderer used only the DC SH term. The current code evaluates SH up to the active degree and passes the `active_mask` down into projection/rasterization.

```python
means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera, mask=mask)

view_dirs = gaussians.means - camera.center
view_dirs = view_dirs / jnp.linalg.norm(view_dirs, axis=-1, keepdims=True)
colors = eval_sh(sh_degree, gaussians.sh_coeffs, view_dirs)
colors = jnp.clip(colors + 0.5, 0.0, 1.0)
```

This change matters because padded inactive entries are otherwise not just wasted compute; they also perturb density statistics and rasterizer load.

---

## 6. Renderer-Health-Gated Training

One of the main recent changes is that the renderer now produces load diagnostics that feed back into scheduling decisions.

```python
extras = {
    "n_interactions": n_interactions,
    "mean_interactions_per_tile": n_interactions / jnp.maximum(num_tiles, 1),
    "max_interactions_per_tile": jnp.max(tile_counts),
    "overflow_tiles": jnp.sum(tile_counts > BLOCK_SIZE),
    "overflow_interactions": jnp.sum(overflow),
    "truncated_tiles": jnp.sum(tile_counts > MAX_TILE_INTERACTIONS),
    "truncated_interactions": jnp.sum(truncated),
    "radius_cap_violations": radius_cap_violations,
}
```

The training loop then uses these signals to decide whether densification should proceed and whether SH degree promotion should be held back.

```python
quality_healthy = (
    block_metrics["overflow_tiles"] <= max_overflow_tiles and
    block_metrics["overflow_interactions"] <= max_overflow_interactions and
    block_metrics["radius_cap_violations"] <= max_radius_cap_violations and
    block_metrics["truncated_tiles"] <= max_truncated_tiles and
    block_metrics["truncated_interactions"] <= max_truncated_interactions
)

densify_enabled = quality_healthy and (block_metrics["truncated_tiles"] == 0.0)
```

This is a genuine difference from the original implementation. Kerbl et al. densify according to geometric and photometric heuristics, but they do not need to regulate training based on a statically budgeted JAX tile pipeline with explicit overflow and truncation thresholds.

The same logic now controls SH promotion:

```python
desired_sh_degree = min(3, curr_iter // 1000)
if sh_promotion_mode == "always":
    next_sh_degree = desired_sh_degree
elif quality_healthy and desired_sh_degree > curr_sh_degree:
    next_sh_degree = curr_sh_degree + 1
else:
    next_sh_degree = curr_sh_degree
```

This preserves the spirit of progressive SH fitting from 3DGS while avoiding promotion when the renderer is already overloaded.

---

## 7. Multi-Device Execution

`train_parallel.py` uses `jax.pmap` for data parallelism, but adaptive density control is not applied independently on every replica. Instead, the code unreplicates one authoritative state, performs clone/split/prune once, and then re-replicates the result.

```python
authoritative_state = unreplicate_tree(curr_state)
authoritative_state = density_step(authoritative_state, density_rng, densify_enabled)
curr_state = replicate_tree(authoritative_state, devices)
```

This is another place where the JAX implementation departs from the original CUDA-oriented setup. The original method is not written around replicated XLA programs, so it does not need an explicit authoritative densification step to keep replicas structurally identical.

Within the per-device training block, gradients and diagnostic summaries are synchronized explicitly:

```python
grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="batch"), grads)
loss = jax.lax.pmean(loss, axis_name="batch")
metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="batch"), metrics)
```

The max-radius update additionally uses `pmax` on masked radii to avoid invalid near-plane projections dominating the statistic.

---

## 8. Comparison to the Original 3DGS Implementation

The table below distinguishes between ideas inherited from Kerbl et al. and implementation changes introduced by `jax-gs`.

| Axis | Kerbl et al. (2023) | Current `jax-gs` |
| :--- | :--- | :--- |
| Training stack | PyTorch + custom CUDA rasterization | Pure JAX + XLA, no custom CUDA kernels |
| Gaussian set representation | Dynamically managed point set | Fixed-capacity padded buffer with `active_mask` |
| Densification mechanics | Clone/split/prune on dynamic tensors | Clone/split/prune via static gather/scatter routing |
| Optimizer consistency during densify | Natural under direct tensor mutation | Explicit optimizer-state reordering required |
| SH color | Progressive SH fitting | Progressive SH fitting with optional health-gated promotion |
| Rasterization | CUDA tile rasterizer | Pure JAX rasterizer plus TPU-specialized path |
| Systems feedback into training | Mainly image/loss-driven | Image/loss-driven plus renderer overflow/truncation diagnostics |
| Multi-device execution | Not formulated as replicated XLA state | `pmap` training with authoritative densify-and-broadcast |

Two clarifications are important:

1. `jax-gs` does not replace the 3DGS algorithm; it re-expresses it under JAX constraints.
2. Several recent fixes are implementation repairs rather than new modeling contributions. Examples include correct optimizer-state permutation, visibility-aware denominators, and masked max-radius accumulation.

---

## 9. Practical Commands

Single-device training:

```bash
python train.py \
  --data_path data/nerf_example_data/nerf_llff_data/fern \
  --images_subdir images_8 \
  --fast_tpu_rasterizer \
  --density_interval 500 \
  --max_gaussians_growth 8 \
  --max_gaussians_cap 200000
```

Multi-device training:

```bash
python train_parallel.py \
  --data_path data/nerf_example_data/nerf_llff_data/fern \
  --images_subdir images_8 \
  --fast_tpu_rasterizer \
  --density_interval 500 \
  --sh_promotion_mode health_gated
```

CPU-side regression tests:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu pytest tests/
```

---

## 10. Conclusion

The current `jax-gs` implementation should be understood as a static-shape systems reinterpretation of 3D Gaussian Splatting rather than a line-by-line port of the original code. The main technical result is that adaptive density control, SH scheduling, and tile-based splatting can be preserved inside JAX, provided that the implementation is reorganized around padded state, explicit masks, block compilation, and renderer-aware control logic.

Relative to Kerbl et al. (2023), the most consequential differences are not the scene representation or objective, but the execution discipline imposed by XLA. Recent updates move the codebase in that direction more rigorously: they restore optimizer correctness after densification, correct the scale of density statistics, prevent inactive slots from leaking into rendering, and regulate training with renderer-health diagnostics. That combination is what makes the current implementation materially more stable than the earlier blog version.
