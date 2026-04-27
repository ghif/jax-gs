import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax_gs.renderer.rasterizer import get_tile_interactions, BLOCK_SIZE
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
    c0_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    c1_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    c2_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    c3_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    T = jnp.ones((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    
    # Pre-generate pixel grid coordinates for this tile
    py, px = jnp.meshgrid(jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32), 
                          jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32), 
                          indexing='ij')
    grid_x = pix_min_x + px
    grid_y = pix_min_y + py

    def cond_fn(state):
        i, _, _, _, _, T = state
        return (i < end_idx) & (jnp.max(T) > 1e-4)

    def body_fn(state):
        i, c0_accum, c1_accum, c2_accum, c3_accum, T = state

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
        
        weight = alpha * T
        c0_accum = c0_accum + weight * c0
        c1_accum = c1_accum + weight * c1
        c2_accum = c2_accum + weight * c2
        c3_accum = c3_accum + weight * c3
        T = T * (1.0 - alpha)

        return i + 1, c0_accum, c1_accum, c2_accum, c3_accum, T

    _, c0_final, c1_final, c2_final, c3_final, final_T = jax.lax.while_loop(
        cond_fn, body_fn, (start_idx, c0_accum, c1_accum, c2_accum, c3_accum, T)
    )

    # Apply background color
    bg0 = background_ref[0]
    bg1 = background_ref[1]
    bg2 = background_ref[2]
    bg3 = background_ref[3]
    
    c0_final = c0_final + final_T * bg0
    c1_final = c1_final + final_T * bg1
    c2_final = c2_final + final_T * bg2
    c3_final = c3_final + final_T * bg3

    final_color = jnp.stack([c0_final, c1_final, c2_final, c3_final], axis=-1)
    out_grid_ref[...] = jnp.nan_to_num(final_color)


def rasterize_kernel_tpu(
    g_means_ref, g_icov_ref, g_ops_ref, g_cols_ref, g_mask_ref, background_ref,
    out_grid_ref,
    *, W, H
):
    """
    Optimized Pallas kernel for TPU (Mosaic).
    Uses pre-gathered attributes and fori_loop for hardware pipelining.
    """
    # Grid mapping
    ty = pl.program_id(0)
    tx = pl.program_id(1)
    
    pix_min_x = (tx * TILE_SIZE).astype(jnp.float32)
    pix_min_y = (ty * TILE_SIZE).astype(jnp.float32)
    
    # Pre-generate pixel grid coordinates for this tile directly as 1D vectors to avoid unsupported reshape
    flat_idx = jnp.arange(TILE_SIZE * TILE_SIZE, dtype=jnp.int32)
    py = (flat_idx // TILE_SIZE).astype(jnp.float32)
    px = (flat_idx % TILE_SIZE).astype(jnp.float32)
    grid_x = pix_min_x + px
    grid_y = pix_min_y + py

    # Initialize accumulators as 1D vectors to perfectly fill TPU registers (8x128)
    c0_accum = jnp.zeros((256,), dtype=jnp.float32)
    c1_accum = jnp.zeros((256,), dtype=jnp.float32)
    c2_accum = jnp.zeros((256,), dtype=jnp.float32)
    c3_accum = jnp.zeros((256,), dtype=jnp.float32)
    T = jnp.ones((256,), dtype=jnp.float32)

    # Use a Python loop to fully unroll the blocks. 
    # This avoids fori_loop overhead and allows the compiler to pipeline everything.
    CHUNK_SIZE = 16
    num_chunks = BLOCK_SIZE // CHUNK_SIZE

    for chunk_idx in range(num_chunks):
        # Load a chunk of 16 Gaussians into vector registers using pl.ds (BLOCK LOAD)
        # This is MUCH faster on TPU than individual scalar loads.
        curr_start = chunk_idx * CHUNK_SIZE
        mu_chunk = g_means_ref[pl.ds(curr_start, CHUNK_SIZE), :]
        icov_chunk = g_icov_ref[pl.ds(curr_start, CHUNK_SIZE), :]
        op_chunk = g_ops_ref[pl.ds(curr_start, CHUNK_SIZE), 0]
        col_chunk = g_cols_ref[pl.ds(curr_start, CHUNK_SIZE), :]
        mask_chunk = g_mask_ref[pl.ds(curr_start, CHUNK_SIZE), 0]

        # Fully unroll Gaussian processing within the chunk
        for l in range(CHUNK_SIZE):
            mu_x = mu_chunk[l, 0]
            mu_y = mu_chunk[l, 1]
            icov_00 = icov_chunk[l, 0]
            icov_01 = icov_chunk[l, 1]
            icov_11 = icov_chunk[l, 3]
            op = op_chunk[l]
            color = col_chunk[l, :]
            mask = mask_chunk[l].astype(bool)

            dx = grid_x - mu_x
            dy = grid_y - mu_y

            # Compute Gaussian influence
            power = -0.5 * (dx * dx * icov_00 + dx * dy * 2.0 * icov_01 + dy * dy * icov_11)
            alpha = jnp.exp(jnp.clip(power, -100.0, 0.0)) * op
            
            # Mask inactive or saturated pixels
            is_active = mask & (power > -10.0) & (grid_x < W) & (grid_y < H) & (T > 1e-4)
            alpha = jnp.where(is_active, jnp.minimum(0.99, alpha), 0.0)

            weight = alpha * T
            
            # Vectorized accumulation across 256 pixels simultaneously
            c0_accum = c0_accum + weight * color[0]
            c1_accum = c1_accum + weight * color[1]
            c2_accum = c2_accum + weight * color[2]
            c3_accum = c3_accum + weight * color[3]
            T = T * (1.0 - alpha)

    # Apply background color to 1D vectors
    bg = background_ref[...] # (4,)
    c0_accum = c0_accum + T * bg[0]
    c1_accum = c1_accum + T * bg[1]
    c2_accum = c2_accum + T * bg[2]
    c3_accum = c3_accum + T * bg[3]

    # Reconstruct 3D output block from channels and reshape for storage
    final_color = jnp.stack([c0_accum, c1_accum, c2_accum, c3_accum], axis=-1)
    final_color = final_color.reshape(TILE_SIZE, TILE_SIZE, 4)

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
    
    if backend == "tpu":
        # Pre-gather into dense blocks for TPU to avoid unaligned HBM access
        def get_tile_data(ty, tx):
            tile_idx = ty * num_tiles_x + tx
            start_idx = tile_boundaries[tile_idx]
            end_idx = tile_boundaries[tile_idx + 1]
            count = end_idx - start_idx
            
            # Use dynamic_slice instead of gather for contiguous blocks from the sorted arrays
            means_block = jax.lax.dynamic_slice(means2D_sorted, (start_idx, 0), (BLOCK_SIZE, 2))
            icov_block = jax.lax.dynamic_slice(inv_cov2D_sorted, (start_idx, 0), (BLOCK_SIZE, 4))
            ops_block = jax.lax.dynamic_slice(opacities_sorted, (start_idx,), (BLOCK_SIZE,))
            cols_block = jax.lax.dynamic_slice(colors_sorted, (start_idx, 0), (BLOCK_SIZE, 4))
            
            local_mask = jnp.arange(BLOCK_SIZE) < count
            
            return (means_block, 
                    icov_block, 
                    ops_block[:, None], 
                    cols_block, 
                    local_mask[:, None].astype(jnp.float32))

        # Vectorize over grid
        grid_y = jnp.arange(num_tiles_y)
        grid_x = jnp.arange(num_tiles_x)
        # Nested vmap for 2D grid
        tile_data_fn = jax.vmap(jax.vmap(get_tile_data, in_axes=(None, 0)), in_axes=(0, None))
        g_means, g_icov, g_ops, g_cols, g_mask = tile_data_fn(grid_y, grid_x)

        in_specs = [
            pl.BlockSpec((None, None, BLOCK_SIZE, 2), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 4), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 1), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 4), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 1), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec() # background
        ]

        out_image = pl.pallas_call(
            partial(rasterize_kernel_tpu, W=float(W), H=float(H)),
            out_shape=out_shape,
            grid=(num_tiles_y, num_tiles_x),
            in_specs=in_specs,
            out_specs=out_specs,
            interpret=is_cpu
        )(g_means, g_icov, g_ops, g_cols, g_mask, background_padded)
    else:
        # GPU path: uses dynamic indexing directly from HBM (allowed in Triton)
        out_image = pl.pallas_call(
            partial(rasterize_kernel_gpu, num_tiles_x=num_tiles_x, W=float(W), H=float(H)),
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
