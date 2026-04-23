import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from typing import Tuple
from functools import partial

TILE_SIZE = 16
BLOCK_SIZE = 192

def rasterize_kernel(
    means2D_ref, inv_cov2D_ref, sig_opacities_ref, colors_ref,
    sorted_gaussian_ids_ref, tile_boundaries_ref, background_ref,
    out_grid_ref,
    *, num_tiles_x, W, H, tile_size
):
    """
    Pallas kernel for rasterizing a single tile.
    Allocates accumulator arrays in VMEM and streams Gaussians from HBM.
    """
    tile_idx = pl.program_id(0)
    
    start_idx = tile_boundaries_ref[tile_idx]
    end_idx = tile_boundaries_ref[tile_idx + 1]
    count = end_idx - start_idx
    
    ty = tile_idx // num_tiles_x
    tx = tile_idx % num_tiles_x
    pix_min_x = tx * tile_size
    pix_min_y = ty * tile_size
    
    # Initialize accumulators (VMEM)
    accum_color = jnp.zeros((tile_size, tile_size, 3), dtype=jnp.float32)
    T = jnp.ones((tile_size, tile_size), dtype=jnp.float32)
    
    # Pixel grid for this tile
    # Using jnp.arange and broadcasting instead of mgrid inside kernel for better compatibility
    ys = jnp.arange(tile_size, dtype=jnp.float32) + pix_min_y
    xs = jnp.arange(tile_size, dtype=jnp.float32) + pix_min_x
    grid_x = xs[None, :]
    grid_y = ys[:, None]

    def loop_body(i, carry):
        accum_color, T = carry
        
        # Stream a single Gaussian (HBM -> VMEM)
        idx = sorted_gaussian_ids_ref[start_idx + i]
        
        mu = means2D_ref[idx]
        icov = inv_cov2D_ref[idx]
        op = sig_opacities_ref[idx, 0]
        col = colors_ref[idx]
        
        dx = grid_x - mu[0]
        dy = grid_y - mu[1]
        
        # Expanded quadratic form
        power = -0.5 * (dx * dx * icov[0, 0] + dx * dy * 2.0 * icov[0, 1] + dy * dy * icov[1, 1])
        
        alpha = jnp.exp(power) * op
        
        # Active mask: Gaussian validity + T threshold + power range + image bounds
        valid = (power > -10.0) & (grid_x < W) & (grid_y < H) & (T > 1e-4) & (i < count)
        alpha = jnp.where(valid, jnp.minimum(0.99, alpha), 0.0)
        
        # Blend
        new_color = accum_color + (alpha * T)[..., None] * col
        new_T = T * (1.0 - alpha)
        
        return new_color, new_T
        
    # Main Gaussian loop
    final_color, final_T = jax.lax.fori_loop(0, BLOCK_SIZE, loop_body, (accum_color, T))
    
    # Apply background
    final_color = final_color + final_T[..., None] * background_ref[:]
    
    # Store result back to HBM
    out_grid_ref[tile_idx, ...] = final_color


def render_tiles_pallas(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                        H, W, tile_size: int = TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0])):
    """
    Render tiles using JAX Pallas.
    """
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = jnp.maximum(det, 1e-6)
    
    inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)
    sig_opacities = jax.nn.sigmoid(opacities)
    
    tile_indices = jnp.arange(num_tiles + 1)
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, tile_indices)

    # Pad sorted_gaussian_ids to prevent out-of-bounds reads in the block loop
    pad_size = BLOCK_SIZE
    padded_gaussian_ids = jnp.pad(sorted_gaussian_ids, (0, pad_size), constant_values=0)

    # Define the output shape
    out_shape = jax.ShapeDtypeStruct((num_tiles, tile_size, tile_size, 3), jnp.float32)
    
    # Use partial to bind static arguments to the kernel
    kernel_fn = partial(rasterize_kernel, 
                        num_tiles_x=num_tiles_x, 
                        W=W, H=H, 
                        tile_size=tile_size)

    # Execute Pallas kernel
    # Enable interpret mode on CPU for debugging/compatibility
    is_cpu = jax.devices()[0].platform == "cpu"
    
    out_grid = pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid=(num_tiles,),
        interpret=is_cpu
    )(
        means2D, inv_cov2D, sig_opacities, colors,
        padded_gaussian_ids, tile_boundaries, background
    )
    
    # Reshape the output to an image
    output_grid = out_grid.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    output_image = output_grid.swapaxes(1, 2).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    
    return output_image[:H, :W, :]
