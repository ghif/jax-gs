import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from typing import Tuple, Optional
from functools import partial

TILE_SIZE = 16

def rasterize_kernel_gpu(
    means2D_T_ref, inv_cov2D_T_ref, opacities_T_ref, colors_T_ref,
    tile_boundaries_ref, background_ref,
    out_grid_ref,
    *, num_tiles_x, W, H
):
    """
    Optimized Pallas kernel for GPU (Triton).
    Uses Structure-of-Arrays (SoA) layout and 2D grid mapping.
    """
    # Grid mapping
    ty = pl.program_id(0)
    tx = pl.program_id(1)
    tile_idx = ty * num_tiles_x + tx
    
    # Dynamic range for this tile
    start_idx = tile_boundaries_ref[tile_idx]
    end_idx = tile_boundaries_ref[tile_idx + 1]
    
    pix_min_x = (tx * TILE_SIZE).astype(jnp.float32)
    pix_min_y = (ty * TILE_SIZE).astype(jnp.float32)
    
    # Initialize accumulators
    accum_color = jnp.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=jnp.float32)
    T = jnp.ones((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    
    # Pre-generate pixel grid coordinates for this tile
    py, px = jnp.meshgrid(jnp.arange(TILE_SIZE, dtype=jnp.float32), 
                          jnp.arange(TILE_SIZE, dtype=jnp.float32), 
                          indexing='ij')
    grid_x = pix_min_x + px
    grid_y = pix_min_y + py

    def cond_fn(state):
        i, _, T = state
        return (i < end_idx) & (jnp.max(T) > 1e-4)

    def body_fn(state):
        i, accum_color, T = state

        # Load attributes from SoA layout
        mu_x = means2D_T_ref[0, i]
        mu_y = means2D_T_ref[1, i]
        icov_00 = inv_cov2D_T_ref[0, i]
        icov_01 = inv_cov2D_T_ref[1, i]
        icov_11 = inv_cov2D_T_ref[3, i]
        op = opacities_T_ref[i]

        dx = grid_x - mu_x
        dy = grid_y - mu_y

        # Compute Gaussian influence
        power = -0.5 * (dx * dx * icov_00 + dx * dy * 2.0 * icov_01 + dy * dy * icov_11)
        
        # Match rasterizer.py logic exactly
        alpha = jnp.exp(jnp.clip(power, -100.0, 0.0)) * op
        is_active = (power > -10.0) & (grid_x < W) & (grid_y < H) & (T > 1e-4)
        alpha = jnp.where(is_active, jnp.minimum(0.99, alpha), 0.0)

        # Load color components
        c0 = colors_T_ref[0, i]
        c1 = colors_T_ref[1, i]
        c2 = colors_T_ref[2, i]
        c3 = colors_T_ref[3, i]
        
        # Alpha blending: Update each channel separately to avoid concatenate/stack primitives
        # Triton lowering has limitations on concatenate/stack of non-singleton dimensions.
        weight = alpha * T
        
        # Use broadcasting to update the 4th dimension without explicit vector construction
        indices = jnp.arange(4)
        accum_color = accum_color + (weight[..., None] * c0) * (indices == 0).astype(jnp.float32)
        accum_color = accum_color + (weight[..., None] * c1) * (indices == 1).astype(jnp.float32)
        accum_color = accum_color + (weight[..., None] * c2) * (indices == 2).astype(jnp.float32)
        accum_color = accum_color + (weight[..., None] * c3) * (indices == 3).astype(jnp.float32)
        
        T = T * (1.0 - alpha)
        
        # Stability check
        accum_color = jnp.nan_to_num(accum_color)
        T = jnp.nan_to_num(T)

        return i + 1, accum_color, T

    _, final_color, final_T = jax.lax.while_loop(cond_fn, body_fn, (start_idx, accum_color, T))

    # Apply background color channel-wise
    bg0 = background_ref[0]
    bg1 = background_ref[1]
    bg2 = background_ref[2]
    bg3 = background_ref[3]
    
    indices = jnp.arange(4)
    final_color = final_color + (final_T[..., None] * bg0) * (indices == 0).astype(jnp.float32)
    final_color = final_color + (final_T[..., None] * bg1) * (indices == 1).astype(jnp.float32)
    final_color = final_color + (final_T[..., None] * bg2) * (indices == 2).astype(jnp.float32)
    final_color = final_color + (final_T[..., None] * bg3) * (indices == 3).astype(jnp.float32)

    out_grid_ref[...] = jnp.nan_to_num(final_color)


def rasterize_kernel_tpu(
    means2D_T_ref, inv_cov2D_T_ref, opacities_T_ref, colors_T_ref,
    tile_boundaries_ref, background_ref,
    out_grid_ref,
    *, num_tiles_x, W, H
):
    """
    Optimized Pallas kernel for TPU (Mosaic).
    Uses chunked streaming to VMEM and tile-level early termination.
    """
    # Grid mapping
    ty = pl.program_id(0)
    tx = pl.program_id(1)
    tile_idx = ty * num_tiles_x + tx
    
    # Dynamic range for this tile
    start_idx = tile_boundaries_ref[tile_idx]
    end_idx = tile_boundaries_ref[tile_idx + 1]
    
    pix_min_x = (tx * TILE_SIZE).astype(jnp.float32)
    pix_min_y = (ty * TILE_SIZE).astype(jnp.float32)
    
    # Initialize accumulators in VMEM
    accum_color = jnp.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=jnp.float32)
    T = jnp.ones((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    
    # Pre-generate pixel grid coordinates for this tile
    py, px = jnp.meshgrid(jnp.arange(TILE_SIZE, dtype=jnp.float32), 
                          jnp.arange(TILE_SIZE, dtype=jnp.float32), 
                          indexing='ij')
    grid_x = pix_min_x + px
    grid_y = pix_min_y + py

    CHUNK_SIZE = 16
    num_gaussians = end_idx - start_idx
    num_chunks = (num_gaussians + CHUNK_SIZE - 1) // CHUNK_SIZE

    def chunk_body(chunk_idx, state):
        accum_color, T = state
        
        def process_chunk(state):
            accum_color, T = state
            curr_start = start_idx + chunk_idx * CHUNK_SIZE
            
            # Load chunks of Gaussian attributes into VMEM using dynamic slicing (pl.ds)
            m2d_chunk = means2D_T_ref[slice(None), pl.ds(curr_start, CHUNK_SIZE)]
            icov_chunk = inv_cov2D_T_ref[slice(None), pl.ds(curr_start, CHUNK_SIZE)]
            op_chunk = opacities_T_ref[pl.ds(curr_start, CHUNK_SIZE)]
            col_chunk = colors_T_ref[slice(None), pl.ds(curr_start, CHUNK_SIZE)]

            def gaussian_body(j, state):
                accum_color, T = state
                i_global = curr_start + j
                
                # Bound check for the last partial chunk
                mask = i_global < end_idx
                
                mu_x = m2d_chunk[0, j]
                mu_y = m2d_chunk[1, j]
                icov_00 = icov_chunk[0, j]
                icov_01 = icov_chunk[1, j]
                icov_11 = icov_chunk[3, j]
                op = op_chunk[j]

                dx = grid_x - mu_x
                dy = grid_y - mu_y

                # Compute Gaussian influence
                power = -0.5 * (dx * dx * icov_00 + dx * dy * 2.0 * icov_01 + dy * dy * icov_11)
                alpha = jnp.exp(jnp.clip(power, -100.0, 0.0)) * op
                
                # Mask inactive or saturated pixels
                is_active = mask & (power > -10.0) & (grid_x < W) & (grid_y < H) & (T > 1e-4)
                alpha = jnp.where(is_active, jnp.minimum(0.99, alpha), 0.0)

                c = col_chunk[:, j]
                weight = alpha * T
                
                # Vectorized alpha blending
                new_accum_color = accum_color + weight[..., None] * c[None, None, :]
                new_T = T * (1.0 - alpha)
                
                return new_accum_color, new_T

            # Process the chunk of 16 Gaussians
            accum_color, T = jax.lax.fori_loop(0, CHUNK_SIZE, gaussian_body, (accum_color, T))
            return jnp.nan_to_num(accum_color), jnp.nan_to_num(T)

        # Tile-level early termination
        return jax.lax.cond(jnp.max(T) >= 1e-4, process_chunk, lambda s: s, (accum_color, T))

    # Main loop over Gaussian chunks
    final_color, final_T = jax.lax.fori_loop(0, num_chunks, chunk_body, (accum_color, T))

    # Apply background color
    bg = background_ref[...]
    final_color = final_color + final_T[..., None] * bg[None, None, :]

    # Store result back to HBM
    out_grid_ref[...] = jnp.nan_to_num(final_color)


def render_tiles_pallas(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                        H, W, tile_size: int = TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0]),
                        backend: str = "gpu"):
    """
    Render tiles using JAX Pallas, compatible with both GPU and TPU.
    """
    assert tile_size == TILE_SIZE
    
    num_tiles_x = int((W + TILE_SIZE - 1) // TILE_SIZE)
    num_tiles_y = int((H + TILE_SIZE - 1) // TILE_SIZE)
    num_tiles = num_tiles_x * num_tiles_y
    
    # Inverse covariances
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = jnp.maximum(det, 1e-6)
    inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)
    sig_opacities = jax.nn.sigmoid(opacities)
    
    # Tile boundaries
    tile_indices = jnp.arange(num_tiles + 1)
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, tile_indices)

    # Padding colors and background
    colors_padded = jnp.concatenate([colors, jnp.zeros((colors.shape[0], 1))], axis=-1)
    background_padded = jnp.concatenate([background, jnp.zeros((1,))])

    # Pre-gather and Transpose
    valid_ids = jnp.where(sorted_gaussian_ids < means2D.shape[0], sorted_gaussian_ids, 0)
    means2D_sorted = means2D[valid_ids]
    inv_cov2D_sorted = inv_cov2D.reshape(-1, 4)[valid_ids]
    opacities_sorted = sig_opacities[valid_ids, 0]
    colors_sorted = colors_padded[valid_ids]

    out_shape = jax.ShapeDtypeStruct(
        (num_tiles_y * TILE_SIZE, num_tiles_x * TILE_SIZE, 4), 
        jnp.float32
    )
    
    # Output BlockSpec mapping
    out_specs = pl.BlockSpec(
        (TILE_SIZE, TILE_SIZE, 4), 
        lambda ty, tx: (ty * TILE_SIZE, tx * TILE_SIZE, 0)
    )

    is_cpu = jax.devices()[0].platform == "cpu"
    kernel = rasterize_kernel_gpu if backend == "gpu" else rasterize_kernel_tpu
    
    # Execute Pallas kernel with 2D grid
    out_image = pl.pallas_call(
        partial(kernel, num_tiles_x=num_tiles_x, W=float(W), H=float(H)),
        out_shape=out_shape,
        grid=(num_tiles_y, num_tiles_x),
        out_specs=out_specs,
        interpret=is_cpu
    )(
        means2D_sorted.T, 
        inv_cov2D_sorted.T, 
        opacities_sorted, 
        colors_sorted.T,
        tile_boundaries, background_padded
    )
    
    # Crop to actual image size and drop the 4th channel
    return jnp.nan_to_num(out_image[:H, :W, :3])
