import jax
import jax.numpy as jnp
from jax_gs.core.gaussians import Gaussians
import optax
import chex
from typing import Tuple

# To use padded arrays, we need a struct to hold the padded state.
@chex.dataclass
class DensityState:
    gaussians: Gaussians
    opt_state: optax.OptState
    active_mask: jnp.ndarray  # (MAX_GAUSSIANS,) bool mask of which are valid
    num_active: jnp.ndarray   # scalar count
    grad_accum: jnp.ndarray   # (MAX_GAUSSIANS,) accumulated 2D positional gradients
    denom: jnp.ndarray        # (MAX_GAUSSIANS,) number of updates accumulated
    max_radii: jnp.ndarray    # (MAX_GAUSSIANS,) max radii seen so far
    
def init_density_state(gaussians: Gaussians, optimizer: optax.GradientTransformation, max_gaussians: int) -> DensityState:
    """Initializes the padded DensityState."""
    n_init = gaussians.means.shape[0]
    
    # Pad Gaussians
    def pad_array(x, val=0.0):
        pad_shape = (max_gaussians - n_init,) + x.shape[1:]
        return jnp.concatenate([x, jnp.full(pad_shape, val, dtype=x.dtype)], axis=0)
    
    padded_means = pad_array(gaussians.means)
    padded_scales = pad_array(gaussians.scales)
    padded_quats = pad_array(gaussians.quaternions, val=0.0)
    # Ensure identity quats for padded ones
    padded_quats = padded_quats.at[n_init:, 0].set(1.0)
    padded_ops = pad_array(gaussians.opacities)
    padded_sh = pad_array(gaussians.sh_coeffs)
    
    padded_gaussians = Gaussians(
        means=padded_means,
        scales=padded_scales,
        quaternions=padded_quats,
        opacities=padded_ops,
        sh_coeffs=padded_sh
    )
    
    active_mask = jnp.arange(max_gaussians) < n_init
    num_active = jnp.array(n_init, dtype=jnp.int32)
    
    opt_state = optimizer.init(padded_gaussians)
    
    grad_accum = jnp.zeros((max_gaussians,), dtype=jnp.float32)
    denom = jnp.zeros((max_gaussians,), dtype=jnp.int32)
    max_radii = jnp.zeros((max_gaussians,), dtype=jnp.float32)
    
    return DensityState(
        gaussians=padded_gaussians,
        opt_state=opt_state,
        active_mask=active_mask,
        num_active=num_active,
        grad_accum=grad_accum,
        denom=denom,
        max_radii=max_radii
    )

def _reset_accumulators(state: DensityState) -> DensityState:
    return state.replace(
        grad_accum=jnp.zeros_like(state.grad_accum),
        denom=jnp.zeros_like(state.denom)
    )

def reorder_opt_state(opt_state, gather_indices):
    """Reorders the optimizer state pytree leaves according to gather_indices."""
    def reorder_leaf(x):
        # We only reorder arrays that match the number of Gaussians
        if hasattr(x, 'ndim') and x.ndim >= 1 and x.shape[0] == gather_indices.shape[0]:
            return x[gather_indices]
        return x
    return jax.tree_util.tree_map(reorder_leaf, opt_state)

def densify_and_prune(
    state: DensityState, 
    rng_key: jax.Array,
    grad_threshold: float = 0.0002, 
    min_opacity: float = 0.005, 
    extent: float = 5.0, # scene radius extent approx
    max_screen_size: int = 20
) -> DensityState:
    """
    Performs densification and pruning within the JAX statically-sized framework.
    """
    g = state.gaussians
    active = state.active_mask
    MAX_GAUSSIANS = active.shape[0]
    
    # 1. PRUNE
    # Opacity condition
    sig_ops = jax.nn.sigmoid(g.opacities[..., 0])
    prune_mask = (sig_ops < min_opacity)
    
    # Size condition (world space)
    scales_act = jnp.exp(g.scales)
    max_scales = jnp.max(scales_act, axis=-1)
    prune_mask = prune_mask | (max_scales > 0.1 * extent)
    
    # Screen size condition
    prune_mask = prune_mask | (state.max_radii > max_screen_size)
    
    prune_mask = prune_mask & active
    active_after_prune = active & (~prune_mask)
    
    # 2. DENSIFY
    avg_grad = jnp.where(state.denom > 0, state.grad_accum / state.denom, 0.0)
    densify_mask = (avg_grad >= grad_threshold) & active_after_prune
    
    split_mask = densify_mask & (max_scales > 0.01 * extent)
    clone_mask = densify_mask & (~split_mask)
    
    num_clones = jnp.sum(clone_mask)
    num_splits = jnp.sum(split_mask)
    total_new = num_clones + num_splits
    
    # Check capacity
    will_be_empty = ~active_after_prune
    available_slots = jnp.sum(will_be_empty)
    can_densify = total_new <= available_slots
    
    clone_mask = jnp.where(can_densify, clone_mask, False)
    split_mask = jnp.where(can_densify, split_mask, False)
    num_clones = jnp.sum(clone_mask)
    num_splits = jnp.sum(split_mask)
    
    # --- Vectorized Routing ---
    # Compact indices for sources and destinations using argsort
    empty_indices = jnp.argsort((~will_be_empty).astype(jnp.int8))
    clone_src_indices = jnp.argsort((~clone_mask).astype(jnp.int8))
    split_src_indices = jnp.argsort((~split_mask).astype(jnp.int8))
    
    idx = jnp.arange(MAX_GAUSSIANS)
    is_clone_dest = idx < num_clones
    is_split_dest = (idx >= num_clones) & (idx < num_clones + num_splits)
    
    # Identify which source to read for each empty slot
    src_to_read = jnp.where(is_clone_dest, clone_src_indices[idx], 0)
    src_to_read = jnp.where(is_split_dest, split_src_indices[idx - num_clones], src_to_read)
    
    new_values = jnp.where(is_clone_dest | is_split_dest, src_to_read, empty_indices)
    
    gather_indices = jnp.arange(MAX_GAUSSIANS)
    gather_indices = gather_indices.at[empty_indices].set(new_values)
    
    # --- Execute Data Movement ---
    new_means = g.means[gather_indices]
    new_scales = g.scales[gather_indices]
    new_quats = g.quaternions[gather_indices]
    new_ops = g.opacities[gather_indices]
    new_sh = g.sh_coeffs[gather_indices]
    
    # Reorder optimizer state moments
    new_opt_state = reorder_opt_state(state.opt_state, gather_indices)
    
    # --- Apply Modifications ---
    new_split_mask = jnp.zeros(MAX_GAUSSIANS, dtype=bool)
    new_split_mask = new_split_mask.at[empty_indices].set(is_split_dest)
    do_split_scale = split_mask | new_split_mask
    
    # Divide scale by 1.6
    new_scales = jnp.where(do_split_scale[:, None], new_scales - jnp.log(1.6), new_scales)
    
    # Break symmetry for splits
    noise = jax.random.normal(rng_key, (MAX_GAUSSIANS, 3)) * jnp.exp(new_scales)
    new_means = jnp.where(do_split_scale[:, None], new_means + noise, new_means)
    
    # --- Update Masks ---
    new_active_add = jnp.zeros(MAX_GAUSSIANS, dtype=bool)
    new_active_add = new_active_add.at[empty_indices].set(is_clone_dest | is_split_dest)
    final_active = active_after_prune | new_active_add
    
    # Create the new Gaussian struct
    new_gaussians = Gaussians(
        means=new_means,
        scales=new_scales,
        quaternions=new_quats,
        opacities=new_ops,
        sh_coeffs=new_sh
    )
    
    # Return reset accumulators
    state = _reset_accumulators(state)
    state = state.replace(gaussians=new_gaussians, opt_state=new_opt_state, active_mask=final_active)
    
    return state
