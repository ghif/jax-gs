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
    accum_color = jnp.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=jnp.float32)
    T = jnp.ones((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    
    # Pre-generate pixel grid coordinates for this tile
    py, px = jnp.meshgrid(jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32), 
                          jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32), 
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
    g_packed_ref, background_ref,
    out_grid_ref,
    *, W, H
):
    """
    Optimized Pallas kernel for TPU (Mosaic).
    Uses packed attributes and jax.lax.scan for maximal hardware pipelining.
    """
    # Grid mapping
    ty = pl.program_id(0)
    tx = pl.program_id(1)
    
    pix_min_x = (tx * TILE_SIZE).astype(jnp.float32)
    pix_min_y = (ty * TILE_SIZE).astype(jnp.float32)
    
    # Initialize accumulators in VMEM
    accum_color = jnp.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=jnp.float32)
    T = jnp.ones((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    
    # Pre-generate pixel grid coordinates for this tile
    py, px = jnp.meshgrid(jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32), 
                          jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32), 
                          indexing='ij')
    grid_x = pix_min_x + px
    grid_y = pix_min_y + py

    # Load packed attribute block into VMEM once to avoid slow scalar Ref gathers
    packed_data = g_packed_ref[...]

    for j in range(BLOCK_SIZE):
        # Static slice (Python integer j) is much faster on TPU than dynamic pl.ds
        packed_val = packed_data[j]
        
        # Unpack the 12-channel vector
        # [mu_x, mu_y, icov_00, icov_01, icov_10, icov_11, op, c0, c1, c2, c3, mask]
        mu_x = packed_val[0]
        mu_y = packed_val[1]
        icov_00 = packed_val[2]
        icov_01 = packed_val[3]
        icov_11 = packed_val[5]
        op = packed_val[6]
        c = packed_val[7:11]
        mask = packed_val[11].astype(bool)

        dx = grid_x - mu_x
        dy = grid_y - mu_y

        # Compute Gaussian influence
        power = -0.5 * (dx * dx * icov_00 + dx * dy * 2.0 * icov_01 + dy * dy * icov_11)
        alpha = jnp.exp(jnp.clip(power, -100.0, 0.0)) * op
        
        # Mask inactive or saturated pixels
        is_active = mask & (power > -10.0) & (grid_x < W) & (grid_y < H) & (T > 1e-4)
        alpha = jnp.where(is_active, jnp.minimum(0.99, alpha), 0.0)

        weight = alpha * T
        
        # Direct vectorized alpha blending (much faster on TPU)
        accum_color = accum_color + weight[..., None] * c[None, None, :]
        T = T * (1.0 - alpha)

    # Apply background color
    bg = background_ref[...]
    accum_color = accum_color + T[..., None] * bg[None, None, :]

    # Store result back to HBM
    out_grid_ref[...] = jnp.nan_to_num(accum_color)


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
            
            gather_indices = jnp.clip(start_idx + jnp.arange(BLOCK_SIZE), 0, means2D_sorted.shape[0] - 1)
            local_mask = (start_idx + jnp.arange(BLOCK_SIZE)) < (start_idx + count)
            
            # Pack all attributes into a single 12-channel array to bypass Scan arg limits
            return jnp.concatenate([
                means2D_sorted[gather_indices],          # 2
                inv_cov2D_sorted[gather_indices],        # 4
                opacities_sorted[gather_indices, None],  # 1
                colors_sorted[gather_indices],           # 4
                local_mask[:, None].astype(jnp.float32)  # 1
            ], axis=-1)

        # Vectorize over grid
        grid_y = jnp.arange(num_tiles_y)
        grid_x = jnp.arange(num_tiles_x)
        # Nested vmap for 2D grid
        tile_data_fn = jax.vmap(jax.vmap(get_tile_data, in_axes=(None, 0)), in_axes=(0, None))
        g_packed = tile_data_fn(grid_y, grid_x)

        in_specs = [
            pl.BlockSpec((None, None, BLOCK_SIZE, 12), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec() # background
        ]

        out_image = pl.pallas_call(
            partial(rasterize_kernel_tpu, W=float(W), H=float(H)),
            out_shape=out_shape,
            grid=(num_tiles_y, num_tiles_x),
            in_specs=in_specs,
            out_specs=out_specs,
            interpret=is_cpu
        )(g_packed, background_padded)
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
