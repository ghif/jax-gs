import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax_gs.renderer.rasterizer import get_tile_interactions, BLOCK_SIZE
from typing import Tuple, Optional
from functools import partial

TILE_SIZE = 16
BLOCK_SIZE = 1024

def rasterize_kernel_gpu(
    means2D_T_ref, inv_cov2D_T_ref, opacities_T_ref, colors_T_ref,
    depths_T_ref, normals_T_ref,
    tile_boundaries_ref, background_ref,
    out_grid_ref,
    *, num_tiles_x, W, H, is_2dgs
):
    """
    Optimized Pallas kernel for GPU (Triton).
    Uses Structure-of-Arrays (SoA) layout and 2D grid mapping.
    """
    ty = pl.program_id(0)
    tx = pl.program_id(1)
    tile_idx = ty * num_tiles_x + tx
    
    start_idx = tile_boundaries_ref[tile_idx]
    end_idx = tile_boundaries_ref[tile_idx + 1]
    
    pix_min_x = (tx * TILE_SIZE).astype(jnp.float32) + 0.5
    pix_min_y = (ty * TILE_SIZE).astype(jnp.float32) + 0.5
    
    c0_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    c1_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    c2_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    T = jnp.ones((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    
    d_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    d2_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    n0_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    n1_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    n2_accum = jnp.zeros((TILE_SIZE, TILE_SIZE), dtype=jnp.float32)
    
    py, px = jnp.meshgrid(jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32), 
                          jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32), 
                          indexing='ij')
    grid_x = pix_min_x + px
    grid_y = pix_min_y + py

    def cond_fn(state):
        i, _, _, _, _, _, _, _, _, _, T = state
        # Early exit if saturated
        return (i < end_idx) & (jnp.max(T) > 1e-4)

    def body_fn(state):
        i, c0_accum, c1_accum, c2_accum, d_accum, d2_accum, n0_accum, n1_accum, n2_accum, _, T = state

        mu_x, mu_y = means2D_T_ref[0, i], means2D_T_ref[1, i]
        icov_00, icov_01, icov_11 = inv_cov2D_T_ref[0, i], inv_cov2D_T_ref[1, i], inv_cov2D_T_ref[3, i]
        op = opacities_T_ref[i]

        dx, dy = grid_x - mu_x, grid_y - mu_y
        power = -0.5 * (dx * dx * icov_00 + dx * dy * 2.0 * icov_01 + dy * dy * icov_11)
        
        alpha = jnp.exp(jnp.clip(power, -100.0, 0.0)) * op
        is_active = (power > -10.0) & (grid_x < W) & (grid_y < H) & (T > 1e-4)
        alpha = jnp.where(is_active, jnp.minimum(0.99, alpha), 0.0)

        weight = alpha * T
        c0_accum = c0_accum + weight * colors_T_ref[0, i]
        c1_accum = c1_accum + weight * colors_T_ref[1, i]
        c2_accum = c2_accum + weight * colors_T_ref[2, i]
        
        if is_2dgs:
            d = depths_T_ref[i]
            d_accum = d_accum + weight * d
            d2_accum = d2_accum + weight * (d * d)
            n0_accum = n0_accum + weight * normals_T_ref[0, i]
            n1_accum = n1_accum + weight * normals_T_ref[1, i]
            n2_accum = n2_accum + weight * normals_T_ref[2, i]

        T = T * (1.0 - alpha)
        return i + 1, c0_accum, c1_accum, c2_accum, d_accum, d2_accum, n0_accum, n1_accum, n2_accum, 0.0, T

    _, c0_final, c1_final, c2_final, d_final, d2_final, n0_final, n1_final, n2_final, _, final_T = jax.lax.while_loop(
        cond_fn, body_fn, (start_idx, c0_accum, c1_accum, c2_accum, d_accum, d2_accum, n0_accum, n1_accum, n2_accum, 0.0, T)
    )

    bg = background_ref[...]
    c0_final, c1_final, c2_final = c0_final + final_T * bg[0], c1_final + final_T * bg[1], c2_final + final_T * bg[2]

    if is_2dgs:
        final_data = jnp.stack([c0_final, c1_final, c2_final, d_final, d2_final, n0_final, n1_final, n2_final, 1.0 - final_T], axis=-1)
    else:
        final_data = jnp.stack([c0_final, c1_final, c2_final, 1.0 - final_T], axis=-1)
    out_grid_ref[...] = jnp.nan_to_num(final_data)


def rasterize_kernel_tpu(
    g_means_ref, g_icov_ref, g_ops_ref, g_cols_ref, g_depths_ref, g_normals_ref, g_mask_ref, 
    background_ref, out_grid_ref,
    *, W, H, is_2dgs
):
    """
    Optimized Pallas kernel for TPU (Mosaic).
    """
    ty, tx = pl.program_id(0), pl.program_id(1)
    pix_min_x = (tx * TILE_SIZE).astype(jnp.float32) + 0.5
    pix_min_y = (ty * TILE_SIZE).astype(jnp.float32) + 0.5
    
    flat_idx = jnp.arange(TILE_SIZE * TILE_SIZE, dtype=jnp.int32)
    grid_x = pix_min_x + (flat_idx % TILE_SIZE).astype(jnp.float32)
    grid_y = pix_min_y + (flat_idx // TILE_SIZE).astype(jnp.float32)

    c0_accum = jnp.zeros((256,), dtype=jnp.float32)
    c1_accum = jnp.zeros((256,), dtype=jnp.float32)
    c2_accum = jnp.zeros((256,), dtype=jnp.float32)
    T = jnp.ones((256,), dtype=jnp.float32)
    
    d_accum = jnp.zeros((256,), dtype=jnp.float32)
    d2_accum = jnp.zeros((256,), dtype=jnp.float32)
    n0_accum = jnp.zeros((256,), dtype=jnp.float32)
    n1_accum = jnp.zeros((256,), dtype=jnp.float32)
    n2_accum = jnp.zeros((256,), dtype=jnp.float32)

    CHUNK_SIZE = 16
    num_chunks = BLOCK_SIZE // CHUNK_SIZE

    for chunk_idx in range(num_chunks):
        curr_start = chunk_idx * CHUNK_SIZE
        mu_chunk = g_means_ref[pl.ds(curr_start, CHUNK_SIZE), :]
        icov_chunk = g_icov_ref[pl.ds(curr_start, CHUNK_SIZE), :]
        op_chunk = g_ops_ref[pl.ds(curr_start, CHUNK_SIZE), 0]
        col_chunk = g_cols_ref[pl.ds(curr_start, CHUNK_SIZE), :]
        mask_chunk = g_mask_ref[pl.ds(curr_start, CHUNK_SIZE), 0]
        
        if is_2dgs:
            depth_chunk = g_depths_ref[pl.ds(curr_start, CHUNK_SIZE), 0]
            norm_chunk = g_normals_ref[pl.ds(curr_start, CHUNK_SIZE), :]

        for l in range(CHUNK_SIZE):
            mu_x, mu_y = mu_chunk[l, 0], mu_chunk[l, 1]
            icov_00, icov_01, icov_11 = icov_chunk[l, 0], icov_chunk[l, 1], icov_chunk[l, 3]
            op, mask = op_chunk[l], mask_chunk[l].astype(bool)

            dx, dy = grid_x - mu_x, grid_y - mu_y
            power = -0.5 * (dx * dx * icov_00 + dx * dy * 2.0 * icov_01 + dy * dy * icov_11)
            alpha = jnp.exp(jnp.clip(power, -100.0, 0.0)) * op
            is_active = mask & (power > -10.0) & (grid_x < W) & (grid_y < H) & (T > 1e-4)
            alpha = jnp.where(is_active, jnp.minimum(0.99, alpha), 0.0)

            weight = alpha * T
            c0_accum = c0_accum + weight * col_chunk[l, 0]
            c1_accum = c1_accum + weight * col_chunk[l, 1]
            c2_accum = c2_accum + weight * col_chunk[l, 2]
            
            if is_2dgs:
                d = depth_chunk[l]
                d_accum = d_accum + weight * d
                d2_accum = d2_accum + weight * (d * d)
                n0_accum = n0_accum + weight * norm_chunk[l, 0]
                n1_accum = n1_accum + weight * norm_chunk[l, 1]
                n2_accum = n2_accum + weight * norm_chunk[l, 2]

            T = T * (1.0 - alpha)

    bg = background_ref[...]
    c0_accum, c1_accum, c2_accum = c0_accum + T * bg[0], c1_accum + T * bg[1], c2_accum + T * bg[2]

    if is_2dgs:
        final_color = jnp.stack([c0_accum, c1_accum, c2_accum, d_accum, d2_accum, n0_accum, n1_accum, n2_accum, 1.0 - T], axis=-1).reshape(TILE_SIZE, TILE_SIZE, 9)
    else:
        final_color = jnp.stack([c0_accum, c1_accum, c2_accum, 1.0 - T], axis=-1).reshape(TILE_SIZE, TILE_SIZE, 4)
    out_grid_ref[...] = jnp.nan_to_num(final_color)


def render_tiles_pallas(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                        H, W, tile_size: int = TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0]),
                        depths=None, normals=None, backend: str = "gpu"):
    """
    Render tiles using JAX Pallas, supporting both 3DGS and 2DGS.
    """
    is_2dgs = depths is not None and normals is not None
    num_tiles_x, num_tiles_y = int((W + TILE_SIZE - 1) // TILE_SIZE), int((H + TILE_SIZE - 1) // TILE_SIZE)
    num_tiles = num_tiles_x * num_tiles_y
    
    det = jnp.maximum(cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2, 1e-6)
    inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)
    sig_opacities = jax.nn.sigmoid(opacities)
    
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, jnp.arange(num_tiles + 1))
    valid_ids = jnp.where(sorted_gaussian_ids < means2D.shape[0], sorted_gaussian_ids, 0)

    means2D_sorted = means2D[valid_ids]
    inv_cov2D_sorted = inv_cov2D.reshape(-1, 4)[valid_ids]
    opacities_sorted = sig_opacities[valid_ids, 0]
    colors_sorted = colors[valid_ids]
    depths_sorted = depths[valid_ids] if is_2dgs else jnp.zeros((valid_ids.shape[0],))
    normals_sorted = normals[valid_ids] if is_2dgs else jnp.zeros((valid_ids.shape[0], 3))

    num_out_channels = 9 if is_2dgs else 4
    out_shape = jax.ShapeDtypeStruct((num_tiles_y * TILE_SIZE, num_tiles_x * TILE_SIZE, num_out_channels), jnp.float32)
    out_specs = pl.BlockSpec((TILE_SIZE, TILE_SIZE, num_out_channels), lambda ty, tx: (ty * TILE_SIZE, tx * TILE_SIZE, 0))

    is_cpu = jax.devices()[0].platform == "cpu"
    
    if backend == "tpu":
        # Vectorized gather for all tiles at once
        all_tile_indices = tile_boundaries[:-1, None] + jnp.arange(BLOCK_SIZE)[None, :]
        all_tile_indices = jnp.clip(all_tile_indices, 0, valid_ids.shape[0] - 1)
        
        g_means = means2D_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 2)
        g_icov = inv_cov2D_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 4)
        g_ops = opacities_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 1)
        g_cols = colors_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 3)
        g_depths = depths_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 1)
        g_normals = normals_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 3)
        
        tile_counts = tile_boundaries[1:] - tile_boundaries[:-1]
        g_mask = (jnp.arange(BLOCK_SIZE)[None, :] < tile_counts[:, None]).astype(jnp.float32).reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 1)

        in_specs = [pl.BlockSpec((None, None, BLOCK_SIZE, 2), lambda ty, tx: (ty, tx, 0, 0)),
                    pl.BlockSpec((None, None, BLOCK_SIZE, 4), lambda ty, tx: (ty, tx, 0, 0)),
                    pl.BlockSpec((None, None, BLOCK_SIZE, 1), lambda ty, tx: (ty, tx, 0, 0)),
                    pl.BlockSpec((None, None, BLOCK_SIZE, 3), lambda ty, tx: (ty, tx, 0, 0)),
                    pl.BlockSpec((None, None, BLOCK_SIZE, 1), lambda ty, tx: (ty, tx, 0, 0)),
                    pl.BlockSpec((None, None, BLOCK_SIZE, 3), lambda ty, tx: (ty, tx, 0, 0)),
                    pl.BlockSpec((None, None, BLOCK_SIZE, 1), lambda ty, tx: (ty, tx, 0, 0)),
                    pl.BlockSpec()]

        out_image = pl.pallas_call(
            partial(rasterize_kernel_tpu, W=float(W), H=float(H), is_2dgs=is_2dgs),
            out_shape=out_shape, grid=(num_tiles_y, num_tiles_x), in_specs=in_specs, out_specs=out_specs, interpret=is_cpu
        )(g_means, g_icov, g_ops, g_cols, g_depths, g_normals, g_mask, background)
    else:
        out_image = pl.pallas_call(
            partial(rasterize_kernel_gpu, num_tiles_x=num_tiles_x, W=float(W), H=float(H), is_2dgs=is_2dgs),
            out_shape=out_shape, grid=(num_tiles_y, num_tiles_x), out_specs=out_specs, interpret=is_cpu
        )(means2D_sorted.T, inv_cov2D_sorted.T, opacities_sorted, colors_sorted.T, depths_sorted, normals_sorted.T, tile_boundaries, background)
    
    if is_2dgs:
        extras = {"depth": out_image[:H, :W, 3:4], "depth_sq": out_image[:H, :W, 4:5], "normals": out_image[:H, :W, 5:8], "accum_weight": out_image[:H, :W, 8:9]}
        return out_image[:H, :W, 0:3], extras
    else:
        return out_image[:H, :W, 0:3], {}
