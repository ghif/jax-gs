import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax_gs.renderer.rasterizer import TILE_SIZE as BASE_TILE_SIZE, render_tiles
from functools import partial
TILE_SIZE = BASE_TILE_SIZE
BLOCK_SIZE = 1024
EPS = 1e-6
POWER_MIN = -100.0
POWER_ACTIVE_THRESHOLD = -10.0
MAX_ALPHA = 0.99

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
def rasterize_backward_kernel_tile(
    g_means_ref, g_icov_ref, g_ops_ref, g_cols_ref, g_valid_ref,
    final_T_ref, dimage_ref, background_ref,
    d_means_ref, d_icov_ref, d_ops_ref, d_cols_ref,
    *, W, H
):
    ty = pl.program_id(0)
    tx = pl.program_id(1)

    pix_min_x = (tx * TILE_SIZE).astype(jnp.float32) + 0.5
    pix_min_y = (ty * TILE_SIZE).astype(jnp.float32) + 0.5

    py, px = jnp.meshgrid(
        jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32),
        jnp.arange(TILE_SIZE, dtype=jnp.int32).astype(jnp.float32),
        indexing="ij",
    )
    grid_x = pix_min_x + px
    grid_y = pix_min_y + py
    in_bounds = (grid_x < W) & (grid_y < H)

    d0 = dimage_ref[:, :, 0]
    d1 = dimage_ref[:, :, 1]
    d2 = dimage_ref[:, :, 2]

    T_next = final_T_ref[...]
    bg = background_ref[...]
    behind0 = jnp.full((TILE_SIZE, TILE_SIZE), bg[0], dtype=jnp.float32)
    behind1 = jnp.full((TILE_SIZE, TILE_SIZE), bg[1], dtype=jnp.float32)
    behind2 = jnp.full((TILE_SIZE, TILE_SIZE), bg[2], dtype=jnp.float32)

    for i in range(BLOCK_SIZE - 1, -1, -1):
        valid = g_valid_ref[i, 0] > 0.5

        mu_x, mu_y = g_means_ref[i, 0], g_means_ref[i, 1]
        icov_00, icov_01, icov_11 = g_icov_ref[i, 0], g_icov_ref[i, 1], g_icov_ref[i, 3]
        op = g_ops_ref[i, 0]
        c0, c1, c2 = g_cols_ref[i, 0], g_cols_ref[i, 1], g_cols_ref[i, 2]

        dx = grid_x - mu_x
        dy = grid_y - mu_y
        power = -0.5 * (dx * dx * icov_00 + 2.0 * dx * dy * icov_01 + dy * dy * icov_11)
        exp_power = jnp.exp(jnp.clip(power, POWER_MIN, 0.0))
        alpha_pre = exp_power * op

        active = valid & in_bounds & (power > POWER_ACTIVE_THRESHOLD)
        alpha = jnp.where(active, jnp.minimum(MAX_ALPHA, alpha_pre), 0.0)
        one_minus_alpha = jnp.maximum(1.0 - alpha, EPS)
        T_i = T_next / one_minus_alpha
        weight = alpha * T_i

        dcol0 = jnp.sum(d0 * weight)
        dcol1 = jnp.sum(d1 * weight)
        dcol2 = jnp.sum(d2 * weight)

        d_alpha = (d0 * (c0 - behind0) + d1 * (c1 - behind1) + d2 * (c2 - behind2)) * T_i
        unclamped = active & (alpha_pre < MAX_ALPHA)
        d_alpha = jnp.where(unclamped, d_alpha, 0.0)

        d_op = jnp.sum(d_alpha * exp_power)

        clip_mask = (power > POWER_MIN) & (power < 0.0)
        d_power = jnp.where(clip_mask, d_alpha * op * exp_power, 0.0)

        dmu_x = jnp.sum(d_power * (dx * icov_00 + dy * icov_01))
        dmu_y = jnp.sum(d_power * (dx * icov_01 + dy * icov_11))
        dic00 = jnp.sum(d_power * (-0.5 * dx * dx))
        dic01 = jnp.sum(d_power * (-dx * dy))
        dic11 = jnp.sum(d_power * (-0.5 * dy * dy))

        d_means_ref[i, 0] = jnp.where(valid, dmu_x, 0.0)
        d_means_ref[i, 1] = jnp.where(valid, dmu_y, 0.0)
        d_icov_ref[i, 0] = jnp.where(valid, dic00, 0.0)
        d_icov_ref[i, 1] = jnp.where(valid, dic01, 0.0)
        d_icov_ref[i, 2] = jnp.where(valid, dic01, 0.0)
        d_icov_ref[i, 3] = jnp.where(valid, dic11, 0.0)
        d_ops_ref[i, 0] = jnp.where(valid, d_op, 0.0)
        d_cols_ref[i, 0] = jnp.where(valid, dcol0, 0.0)
        d_cols_ref[i, 1] = jnp.where(valid, dcol1, 0.0)
        d_cols_ref[i, 2] = jnp.where(valid, dcol2, 0.0)

        behind0 = alpha * c0 + (1.0 - alpha) * behind0
        behind1 = alpha * c1 + (1.0 - alpha) * behind1
        behind2 = alpha * c2 + (1.0 - alpha) * behind2
        T_next = T_i

def _invert_covariance_2d(cov2D):
    det = jnp.maximum(cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2, EPS)
    return jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)

def _prepare_sorted_inputs(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, depths, normals, is_2dgs):
    num_tiles_x = int((W + TILE_SIZE - 1) // TILE_SIZE)
    num_tiles_y = int((H + TILE_SIZE - 1) // TILE_SIZE)
    num_tiles = num_tiles_x * num_tiles_y

    inv_cov2D = _invert_covariance_2d(cov2D)
    sig_opacities = jax.nn.sigmoid(opacities)
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, jnp.arange(num_tiles + 1))
    valid_ids = jnp.where(sorted_gaussian_ids < means2D.shape[0], sorted_gaussian_ids, 0)

    means2D_sorted = means2D[valid_ids]
    inv_cov2D_sorted = inv_cov2D.reshape(-1, 4)[valid_ids]
    opacities_sorted = sig_opacities[valid_ids, 0]
    colors_sorted = colors[valid_ids]
    depths_sorted = depths[valid_ids] if is_2dgs else jnp.zeros((valid_ids.shape[0],), dtype=jnp.float32)
    normals_sorted = normals[valid_ids] if is_2dgs else jnp.zeros((valid_ids.shape[0], 3), dtype=jnp.float32)

    return {
        "num_tiles_x": num_tiles_x,
        "num_tiles_y": num_tiles_y,
        "num_tiles": num_tiles,
        "tile_boundaries": tile_boundaries,
        "valid_ids": valid_ids,
        "means2D_sorted": means2D_sorted,
        "inv_cov2D_sorted": inv_cov2D_sorted,
        "opacities_sorted": opacities_sorted,
        "colors_sorted": colors_sorted,
        "depths_sorted": depths_sorted,
        "normals_sorted": normals_sorted,
    }

def _run_forward_pallas(prepared, H, W, background, is_2dgs, backend):
    num_tiles_x = prepared["num_tiles_x"]
    num_tiles_y = prepared["num_tiles_y"]
    tile_boundaries = prepared["tile_boundaries"]
    means2D_sorted = prepared["means2D_sorted"]
    inv_cov2D_sorted = prepared["inv_cov2D_sorted"]
    opacities_sorted = prepared["opacities_sorted"]
    colors_sorted = prepared["colors_sorted"]
    depths_sorted = prepared["depths_sorted"]
    normals_sorted = prepared["normals_sorted"]

    num_out_channels = 9 if is_2dgs else 4
    out_shape = jax.ShapeDtypeStruct((num_tiles_y * TILE_SIZE, num_tiles_x * TILE_SIZE, num_out_channels), jnp.float32)
    out_specs = pl.BlockSpec((TILE_SIZE, TILE_SIZE, num_out_channels), lambda ty, tx: (ty * TILE_SIZE, tx * TILE_SIZE, 0))
    is_cpu = jax.devices()[0].platform == "cpu"

    if backend == "tpu":
        all_tile_indices = tile_boundaries[:-1, None] + jnp.arange(BLOCK_SIZE)[None, :]
        max_idx = jnp.maximum(prepared["valid_ids"].shape[0] - 1, 0)
        all_tile_indices = jnp.clip(all_tile_indices, 0, max_idx)

        g_means = means2D_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 2)
        g_icov = inv_cov2D_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 4)
        g_ops = opacities_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 1)
        g_cols = colors_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 3)
        g_depths = depths_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 1)
        g_normals = normals_sorted[all_tile_indices].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 3)

        tile_counts = tile_boundaries[1:] - tile_boundaries[:-1]
        g_mask = (jnp.arange(BLOCK_SIZE)[None, :] < tile_counts[:, None]).astype(jnp.float32).reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 1)

        in_specs = [
            pl.BlockSpec((None, None, BLOCK_SIZE, 2), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 4), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 1), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 3), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 1), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 3), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec((None, None, BLOCK_SIZE, 1), lambda ty, tx: (ty, tx, 0, 0)),
            pl.BlockSpec(),
        ]

        return pl.pallas_call(
            partial(rasterize_kernel_tpu, W=float(W), H=float(H), is_2dgs=is_2dgs),
            out_shape=out_shape,
            grid=(num_tiles_y, num_tiles_x),
            in_specs=in_specs,
            out_specs=out_specs,
            interpret=is_cpu,
        )(g_means, g_icov, g_ops, g_cols, g_depths, g_normals, g_mask, background)

    return pl.pallas_call(
        partial(rasterize_kernel_gpu, num_tiles_x=num_tiles_x, W=float(W), H=float(H), is_2dgs=is_2dgs),
        out_shape=out_shape,
        grid=(num_tiles_y, num_tiles_x),
        out_specs=out_specs,
        interpret=is_cpu,
    )(
        means2D_sorted.T,
        inv_cov2D_sorted.T,
        opacities_sorted,
        colors_sorted.T,
        depths_sorted,
        normals_sorted.T,
        tile_boundaries,
        background,
    )

def _render_tiles_reference_grad(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, d_image):
    def ref_fn(m, c, o, col, bg):
        return render_tiles(m, c, o, col, sorted_tile_ids, sorted_gaussian_ids, H, W, TILE_SIZE, bg)
    _, pullback = jax.vjp(ref_fn, means2D, cov2D, opacities, colors, background)
    return pullback(d_image)

def _run_backward_kernel_tiles(tile_means, tile_icov, tile_ops, tile_cols, tile_valid, final_T_pad, d_image_pad, background, H, W):
    num_tiles_y, num_tiles_x = tile_means.shape[0], tile_means.shape[1]
    shape_means = jax.ShapeDtypeStruct(tile_means.shape, jnp.float32)
    shape_icov = jax.ShapeDtypeStruct(tile_icov.shape, jnp.float32)
    shape_ops = jax.ShapeDtypeStruct(tile_ops.shape, jnp.float32)
    shape_cols = jax.ShapeDtypeStruct(tile_cols.shape, jnp.float32)

    block_slot2 = pl.BlockSpec((None, None, BLOCK_SIZE, 2), lambda ty, tx: (ty, tx, 0, 0))
    block_slot4 = pl.BlockSpec((None, None, BLOCK_SIZE, 4), lambda ty, tx: (ty, tx, 0, 0))
    block_slot1 = pl.BlockSpec((None, None, BLOCK_SIZE, 1), lambda ty, tx: (ty, tx, 0, 0))
    block_slot3 = pl.BlockSpec((None, None, BLOCK_SIZE, 3), lambda ty, tx: (ty, tx, 0, 0))

    in_specs = [
        block_slot2,
        block_slot4,
        block_slot1,
        block_slot3,
        block_slot1,
        pl.BlockSpec((TILE_SIZE, TILE_SIZE), lambda ty, tx: (ty * TILE_SIZE, tx * TILE_SIZE)),
        pl.BlockSpec((TILE_SIZE, TILE_SIZE, 3), lambda ty, tx: (ty * TILE_SIZE, tx * TILE_SIZE, 0)),
        pl.BlockSpec(),
    ]

    return pl.pallas_call(
        partial(rasterize_backward_kernel_tile, W=float(W), H=float(H)),
        out_shape=(shape_means, shape_icov, shape_ops, shape_cols),
        out_specs=(block_slot2, block_slot4, block_slot1, block_slot3),
        in_specs=in_specs,
        grid=(num_tiles_y, num_tiles_x),
        interpret=False,
    )(tile_means, tile_icov, tile_ops, tile_cols, tile_valid, final_T_pad, d_image_pad, background)

def _segment_reduce(values, flat_ids, flat_valid, num_points):
    sentinel = jnp.int32(num_points)
    safe_ids = jnp.where(flat_valid, flat_ids, sentinel)
    reduced = jax.ops.segment_sum(values, safe_ids, num_segments=num_points + 1)
    return reduced[:num_points]

def _render_tiles_pallas_backward_3dgs(means2D, cov2D, opacities, colors, valid_ids, tile_boundaries, H, W, background, final_T, d_image):
    num_points = means2D.shape[0]
    num_tiles_x = int((W + TILE_SIZE - 1) // TILE_SIZE)
    num_tiles_y = int((H + TILE_SIZE - 1) // TILE_SIZE)

    inv_cov2D = _invert_covariance_2d(cov2D)
    sig_opacities = jax.nn.sigmoid(opacities)[:, 0]

    tile_offsets = tile_boundaries[:-1, None] + jnp.arange(BLOCK_SIZE)[None, :]
    max_idx = jnp.maximum(valid_ids.shape[0] - 1, 0)
    tile_indices = jnp.clip(tile_offsets, 0, max_idx)
    tile_valid = tile_offsets < tile_boundaries[1:, None]
    tile_gids = valid_ids[tile_indices]

    tile_means = means2D[tile_gids].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 2)
    tile_icov = inv_cov2D.reshape(-1, 4)[tile_gids].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 4)
    tile_ops = sig_opacities[tile_gids].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 1)
    tile_cols = colors[tile_gids].reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 3)
    tile_valid_f = tile_valid.astype(jnp.float32).reshape(num_tiles_y, num_tiles_x, BLOCK_SIZE, 1)

    pad_h = num_tiles_y * TILE_SIZE - H
    pad_w = num_tiles_x * TILE_SIZE - W
    final_T_pad = jnp.pad(final_T, ((0, pad_h), (0, pad_w)), constant_values=1.0)
    d_image_pad = jnp.pad(d_image, ((0, pad_h), (0, pad_w), (0, 0)))

    d_means_tile, d_icov_tile, d_ops_tile, d_cols_tile = _run_backward_kernel_tiles(
        tile_means, tile_icov, tile_ops, tile_cols, tile_valid_f, final_T_pad, d_image_pad, background, H, W
    )

    flat_ids = tile_gids.reshape(-1)
    flat_valid = tile_valid.reshape(-1)
    d_means_flat = d_means_tile.reshape(-1, 2)
    d_icov_flat = d_icov_tile.reshape(-1, 4)
    d_ops_flat = d_ops_tile.reshape(-1, 1)
    d_cols_flat = d_cols_tile.reshape(-1, 3)

    d_means = _segment_reduce(d_means_flat, flat_ids, flat_valid, num_points)
    d_icov = _segment_reduce(d_icov_flat, flat_ids, flat_valid, num_points)
    d_ops_sig = _segment_reduce(d_ops_flat, flat_ids, flat_valid, num_points)
    d_colors = _segment_reduce(d_cols_flat, flat_ids, flat_valid, num_points)

    d_inv_cov2D = d_icov.reshape(num_points, 2, 2)
    _, pullback_cov = jax.vjp(_invert_covariance_2d, cov2D)
    d_cov2D = pullback_cov(d_inv_cov2D)[0]

    sig_full = jax.nn.sigmoid(opacities)
    d_opacities = d_ops_sig * sig_full * (1.0 - sig_full)

    return d_means, d_cov2D, d_opacities, d_colors

def _render_tiles_pallas_3dgs_impl(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, backend):
    prepared = _prepare_sorted_inputs(
        means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, None, None, False
    )
    out_image = _run_forward_pallas(prepared, H, W, background, False, backend)
    image = out_image[:H, :W, 0:3]
    final_T = 1.0 - out_image[:H, :W, 3]
    return image, final_T, prepared["tile_boundaries"], prepared["valid_ids"]

def _render_tiles_pallas_3dgs_bwd_impl(res, g_image, backend):
    means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, final_T, tile_boundaries, valid_ids = res
    platform = jax.devices()[0].platform
    use_reference = (backend == "gpu" and platform != "gpu") or (backend == "tpu" and platform != "tpu")

    if use_reference:
        d_means, d_cov, d_opacities, d_colors, d_background = _render_tiles_reference_grad(
            means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, g_image
        )
    else:
        d_means, d_cov, d_opacities, d_colors = _render_tiles_pallas_backward_3dgs(
            means2D, cov2D, opacities, colors, valid_ids, tile_boundaries, H, W, background, final_T, g_image
        )
        d_background = jnp.sum(g_image * final_T[:, :, None], axis=(0, 1))

    return (
        d_means,
        d_cov,
        d_opacities,
        d_colors,
        None,
        None,
        None,
        None,
        d_background,
    )

@jax.custom_vjp
def _render_tiles_pallas_3dgs_gpu(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background):
    image, _, _, _ = _render_tiles_pallas_3dgs_impl(
        means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, backend="gpu"
    )
    return image

def _render_tiles_pallas_3dgs_gpu_fwd(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background):
    image, final_T, tile_boundaries, valid_ids = _render_tiles_pallas_3dgs_impl(
        means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, backend="gpu"
    )
    res = (
        means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids,
        H, W, background, final_T, tile_boundaries, valid_ids
    )
    return image, res

def _render_tiles_pallas_3dgs_gpu_bwd(res, g_image):
    return _render_tiles_pallas_3dgs_bwd_impl(res, g_image, backend="gpu")

_render_tiles_pallas_3dgs_gpu.defvjp(_render_tiles_pallas_3dgs_gpu_fwd, _render_tiles_pallas_3dgs_gpu_bwd)

@jax.custom_vjp
def _render_tiles_pallas_3dgs_tpu(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background):
    image, _, _, _ = _render_tiles_pallas_3dgs_impl(
        means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, backend="tpu"
    )
    return image

def _render_tiles_pallas_3dgs_tpu_fwd(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background):
    image, final_T, tile_boundaries, valid_ids = _render_tiles_pallas_3dgs_impl(
        means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, backend="tpu"
    )
    res = (
        means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids,
        H, W, background, final_T, tile_boundaries, valid_ids
    )
    return image, res

def _render_tiles_pallas_3dgs_tpu_bwd(res, g_image):
    return _render_tiles_pallas_3dgs_bwd_impl(res, g_image, backend="tpu")

_render_tiles_pallas_3dgs_tpu.defvjp(_render_tiles_pallas_3dgs_tpu_fwd, _render_tiles_pallas_3dgs_tpu_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 10, 11, 12))
def render_tiles_pallas(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                        H, W, tile_size: int = TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0]),
                        depths=None, normals=None, backend: str = "gpu"):
    """
    Render tiles using JAX Pallas, supporting both 3DGS and 2DGS.
    This is the custom_vjp wrapper.
    """
    image, extras, _ = _render_tiles_pallas_impl(
        means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids,
        H, W, tile_size, background, depths, normals, backend
    )
    return image, extras


def render_tiles_pallas_fwd(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                            H, W, tile_size, background, depths, normals, backend):
    image, extras, internal_res = _render_tiles_pallas_impl(
        means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids,
        H, W, tile_size, background, depths, normals, backend
    )
    
    # residuals for backward
    res = (means2D, cov2D, opacities, colors, background, extras, internal_res)
    return (image, extras), res


def render_tiles_pallas_bwd(sorted_tile_ids, sorted_gaussian_ids, H, W, tile_size, depths, normals, backend, res, grad_out):
    (means2D, cov2D, opacities, colors, background, extras, internal_res) = res
    
    grad_image, grad_extras = grad_out
    is_2dgs = depths is not None and normals is not None
    
    if is_2dgs:
        # Fallback to zero for now for 2DGS
        return (jnp.zeros_like(means2D), jnp.zeros_like(cov2D), jnp.zeros_like(opacities), 
                jnp.zeros_like(colors), jnp.zeros_like(background))

    # For 3DGS, use the optimized Pallas backward implementation
    final_T = internal_res["final_T"]
    tile_boundaries = internal_res["tile_boundaries"]
    valid_ids = internal_res["valid_ids"]
    
    res_3dgs = (means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids,
                H, W, background, final_T, tile_boundaries, valid_ids)
    
    grads = _render_tiles_pallas_3dgs_bwd_impl(res_3dgs, grad_image, backend)
    
    # grads is (d_means, d_cov, d_opacities, d_colors, None, None, None, None, d_background)
    return (grads[0], grads[1], grads[2], grads[3], grads[8])

render_tiles_pallas.defvjp(render_tiles_pallas_fwd, render_tiles_pallas_bwd)


def _render_tiles_pallas_impl(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                             H, W, tile_size: int = TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0]),
                             depths=None, normals=None, backend: str = "gpu"):
    """
    Internal implementation of Pallas rendering. Returns (image, extras, residuals).
    """
    is_2dgs = depths is not None and normals is not None
    if is_2dgs:
        prepared = _prepare_sorted_inputs(
            means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, depths, normals, True
        )
        out_image = _run_forward_pallas(prepared, H, W, background, True, backend)
        extras = {
            "depth": out_image[:H, :W, 3:4],
            "depth_sq": out_image[:H, :W, 4:5],
            "normals": out_image[:H, :W, 5:8],
            "accum_weight": out_image[:H, :W, 8:9],
        }
        res = {
            "final_T": 1.0 - out_image[:H, :W, 8],
            "tile_boundaries": prepared["tile_boundaries"],
            "valid_ids": prepared["valid_ids"]
        }
        return out_image[:H, :W, 0:3], extras, res

    if backend == "tpu":
        image, final_T, tile_boundaries, valid_ids = _render_tiles_pallas_3dgs_impl(
            means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, backend="tpu"
        )
    else:
        image, final_T, tile_boundaries, valid_ids = _render_tiles_pallas_3dgs_impl(
            means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background, backend="gpu"
        )
    
    res = {
        "final_T": final_T,
        "tile_boundaries": tile_boundaries,
        "valid_ids": valid_ids
    }
    return image, {}, res
