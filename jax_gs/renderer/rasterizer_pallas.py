import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from typing import Tuple
from functools import partial

TILE_SIZE = 16

def rasterize_kernel(
    means2D_ref, inv_cov2D_ref, sig_opacities_ref, colors_ref,
    sorted_gaussian_ids_ref, tile_boundaries_ref, background_ref,
    out_grid_ref,
    *, num_tiles_x, W, H, tile_size
):
    """
    Optimized Pallas kernel for GPU (Triton).
    Uses dynamic loop bounds and early ray termination.
    """
    # Grid mapping
    tile_x = pl.program_id(0)
    tile_y = pl.program_id(1)
    tile_idx = tile_y * num_tiles_x + tile_x
    
    # Dynamic range for this tile
    start_idx = tile_boundaries_ref[tile_idx]
    end_idx = tile_boundaries_ref[tile_idx + 1]
    
    pix_min_x = tile_x * tile_size
    pix_min_y = tile_y * tile_size
    
    # Initialize accumulators
    accum_color = jnp.zeros((tile_size, tile_size, 3), dtype=jnp.float32)
    T = jnp.ones((tile_size, tile_size), dtype=jnp.float32)
    
    # Pre-generate pixel grid coordinates for this tile
    py, px = jnp.meshgrid(jnp.arange(tile_size), jnp.arange(tile_size), indexing='ij')
    grid_x = pix_min_x + px.astype(jnp.float32)
    grid_y = pix_min_y + py.astype(jnp.float32)

    # Use while_loop for early termination and dynamic bounds (Optimized for GPU)
    def cond_fn(state):
        i, _, T = state
        # Early termination: stop if all pixels in tile are opaque or no more Gaussians
        return (i < end_idx) & (jnp.max(T) > 1e-4)

    def body_fn(state):
        i, accum_color, T = state
        
        # Load Gaussian index
        idx = sorted_gaussian_ids_ref[i]
        
        # Load attributes from global memory (HBM)
        mu = means2D_ref[idx]
        icov = inv_cov2D_ref[idx]
        op = sig_opacities_ref[idx, 0]
        col = colors_ref[idx]
        
        dx = grid_x - mu[0]
        dy = grid_y - mu[1]
        
        # Compute Gaussian influence
        power = -0.5 * (dx * dx * icov[0, 0] + dx * dy * 2.0 * icov[0, 1] + dy * dy * icov[1, 1])
        alpha = jnp.exp(power) * op
        
        # Visibility check
        valid = (power > -10.0) & (grid_x < W) & (grid_y < H)
        alpha = jnp.where(valid, jnp.minimum(0.99, alpha), 0.0)
        
        # Alpha blending
        new_color = accum_color + (alpha * T)[..., None] * col
        new_T = T * (1.0 - alpha)
        
        return i + 1, new_color, new_T

    _, final_color, final_T = jax.lax.while_loop(cond_fn, body_fn, (start_idx, accum_color, T))
    
    # Apply background color
    bg = background_ref[:]
    final_color = final_color + final_T[..., None] * bg
    
    # Write to output (BlockSpec handles the offset)
    out_grid_ref[...] = final_color


def render_tiles_pallas(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                        H, W, tile_size: int = TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0])):
    """
    Render tiles using JAX Pallas, optimized for GPU.
    """
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    # Compute inverse covariances
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = jnp.maximum(det, 1e-6)
    
    inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)
    sig_opacities = jax.nn.sigmoid(opacities)
    
    # Compute tile boundaries
    tile_indices = jnp.arange(num_tiles + 1)
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, tile_indices)

    # Define the output shape as a padded image (to simplify BlockSpec mapping)
    out_shape = jax.ShapeDtypeStruct(
        (num_tiles_y * tile_size, num_tiles_x * tile_size, 3), 
        jnp.float32
    )
    
    # Output BlockSpec mapping: (tx, ty) -> (ty*tile_size, tx*tile_size)
    out_specs = pl.BlockSpec(
        (tile_size, tile_size, 3), 
        lambda tx, ty: (ty * tile_size, tx * tile_size, 0)
    )

    kernel_fn = partial(rasterize_kernel, 
                        num_tiles_x=num_tiles_x, 
                        W=W, H=H, 
                        tile_size=tile_size)

    # Use interpret mode on CPU for debugging
    is_cpu = jax.devices()[0].platform == "cpu"
    
    # Execute Pallas kernel with 2D grid
    out_image = pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid=(num_tiles_x, num_tiles_y),
        out_specs=out_specs,
        interpret=is_cpu
    )(
        means2D, inv_cov2D, sig_opacities, colors,
        sorted_gaussian_ids, tile_boundaries, background
    )
    
    # Crop to actual image size
    return out_image[:H, :W, :]
