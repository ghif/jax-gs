import mlx.core as mx
import numpy as np
from typing import Tuple

# Standard Constants
TILE_SIZE = 16
BLOCK_SIZE = 192  

def get_tile_interactions(means2D, radii, valid_mask, depths, H, W, tile_size: int = TILE_SIZE):
    """
    Generate tile interactions using MLX.
    """
    num_points = means2D.shape[0]
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    
    min_x = mx.clip((means2D[:, 0] - radii), 0, W - 1)
    max_x = mx.clip((means2D[:, 0] + radii), 0, W - 1)
    min_y = mx.clip((means2D[:, 1] - radii), 0, H - 1)
    max_y = mx.clip((means2D[:, 1] + radii), 0, H - 1)
    
    tile_min_x = (min_x // tile_size).astype(mx.int32)
    tile_max_x = (max_x // tile_size).astype(mx.int32)
    tile_min_y = (min_y // tile_size).astype(mx.int32)
    tile_max_y = (max_y // tile_size).astype(mx.int32)
    
    # Filter points completely outside image
    on_screen = (means2D[:, 0] + radii > 0) & (means2D[:, 0] - radii < W) & \
                (means2D[:, 1] + radii > 0) & (means2D[:, 1] - radii < H)
    
    valid_mask = valid_mask & on_screen & (tile_max_x >= tile_min_x) & (tile_max_y >= tile_min_y)
    
    # Tile offset grid
    OFFSET_SIZE = 8
    off_y, off_x = mx.meshgrid(mx.arange(OFFSET_SIZE), mx.arange(OFFSET_SIZE), indexing='ij')
    off_x = off_x.flatten()
    off_y = off_y.flatten()

    # Use broadcasting for interaction generation
    abs_x = tile_min_x[:, None] + off_x[None, :]
    abs_y = tile_min_y[:, None] + off_y[None, :]
    
    # MLX boolean mask
    in_range = (abs_x <= tile_max_x[:, None]) & (abs_y <= tile_max_y[:, None]) & valid_mask[:, None]
    
    all_tile_ids = abs_y * num_tiles_x + abs_x
    all_tile_ids = mx.where(in_range, all_tile_ids, -1)
    
    all_gaussian_ids = mx.broadcast_to(mx.arange(num_points)[:, None], all_tile_ids.shape)
    
    flat_tile_ids = all_tile_ids.reshape(-1)
    flat_gaussian_ids = all_gaussian_ids.reshape(-1)
    flat_depths = mx.broadcast_to(depths[:, None], all_tile_ids.shape).reshape(-1)
    
    valid_interactions = flat_tile_ids != -1
    
    # Pack-Sort for MLX: Match JAX's [TileID: 18 bits] [Depth: 13 bits] exactly
    # JAX uses bitcast(f32, i32) >> 18 for depth
    num_tiles_total = num_tiles_x * num_tiles_y
    sort_tile_ids = mx.where(valid_interactions, flat_tile_ids, num_tiles_total).astype(mx.int32)
    
    # Use Numpy for exact bitcast simulation (non-differentiable, but interaction gen isn't)
    depths_np = np.array(flat_depths)
    depth_i32 = depths_np.view(np.int32)
    depth_quant_np = depth_i32 >> (31 - 13)
    depth_quant = mx.array(depth_quant_np, dtype=mx.int32)
    
    # Construct 31-bit Key using uint32 to avoid sign issues
    key = (sort_tile_ids.astype(mx.uint32) << 13) | depth_quant.astype(mx.uint32)
    sort_indices = mx.argsort(key)
    
    # Correct: Use the CLAMPED IDs for the sorted result so searchsorted works
    sorted_tile_ids = sort_tile_ids[sort_indices]
    sorted_gaussian_ids = flat_gaussian_ids[sort_indices].astype(mx.int32)
    
    total_interactions = sorted_tile_ids.shape[0]
    padded_size = max(total_interactions, BLOCK_SIZE)
    
    if padded_size > total_interactions:
        pad_len = padded_size - total_interactions
        pad_tile_ids = mx.concatenate([sorted_tile_ids, mx.full((pad_len,), num_tiles_total, dtype=mx.int32)])
        pad_gaussian_ids = mx.concatenate([sorted_gaussian_ids, mx.zeros((pad_len,), dtype=mx.int32)])
    else:
        pad_tile_ids = sorted_tile_ids
        pad_gaussian_ids = sorted_gaussian_ids
    
    return pad_tile_ids, pad_gaussian_ids, valid_interactions.sum().item()

def render_tiles(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                 H, W, tile_size: int = TILE_SIZE, background=mx.array([0.0, 0.0, 0.0])):
    """
    Render tiles using MLX.
    """
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = mx.maximum(det, 1e-6)
    
    # Exact JAX Inverse Covariance Indexing
    inv_cov2D = mx.stack([
        mx.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        mx.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)
    
    sig_opacities = mx.sigmoid(opacities)

    # Tile boundaries
    sorted_tile_ids_np = np.array(sorted_tile_ids)
    tile_indices = np.arange(num_tiles + 1)
    tile_boundaries = mx.array(np.searchsorted(sorted_tile_ids_np, tile_indices))
    
    # Pre-gather all parameters to avoid closure issues and improve performance
    t_means_all = mx.take(means2D, sorted_gaussian_ids, axis=0)
    t_inv_cov_all = mx.take(inv_cov2D, sorted_gaussian_ids, axis=0)
    t_ops_all = mx.sigmoid(opacities)[sorted_gaussian_ids, 0]
    t_cols_all = colors[sorted_gaussian_ids]
    # All pixels in image
    y_full = mx.arange(num_tiles_y * tile_size).astype(mx.float32)
    x_full = mx.arange(num_tiles_x * tile_size).astype(mx.float32)
    gy_full, gx_full = mx.meshgrid(y_full, x_full, indexing='ij')
    
    # Reshape to (num_tiles_y, 16, num_tiles_x, 16)
    gy_tiles = gy_full.reshape(num_tiles_y, tile_size, num_tiles_x, tile_size)
    gx_tiles = gx_full.reshape(num_tiles_y, tile_size, num_tiles_x, tile_size)
    
    # TY, PY, TX, PX -> TY, TX, PY, PX
    gy_tiles = gy_tiles.transpose(0, 2, 1, 3).reshape(num_tiles, tile_size, tile_size)
    gx_tiles = gx_tiles.transpose(0, 2, 1, 3).reshape(num_tiles, tile_size, tile_size)

    def rasterize_single_tile(start_idx, end_idx, grid_y, grid_x):
        # Pull interaction parameters for this tile
        count = end_idx - start_idx
        gather_indices = mx.clip(start_idx + mx.arange(BLOCK_SIZE), 0, t_means_all.shape[0] - 1)
        t_means = mx.take(t_means_all, gather_indices, axis=0)
        t_inv_cov = mx.take(t_inv_cov_all, gather_indices, axis=0)
        t_ops = mx.take(t_ops_all, gather_indices, axis=0)
        t_cols = mx.take(t_cols_all, gather_indices, axis=0)
        
        local_mask = mx.arange(BLOCK_SIZE) < count

        # Power calculation
        dx = mx.expand_dims(grid_x, 0) - mx.expand_dims(mx.expand_dims(t_means[:, 0], -1), -1)
        dy = mx.expand_dims(grid_y, 0) - mx.expand_dims(mx.expand_dims(t_means[:, 1], -1), -1)
        
        icov00 = mx.expand_dims(mx.expand_dims(t_inv_cov[:, 0, 0], -1), -1)
        icov01 = mx.expand_dims(mx.expand_dims(t_inv_cov[:, 0, 1], -1), -1)
        icov10 = mx.expand_dims(mx.expand_dims(t_inv_cov[:, 1, 0], -1), -1)
        icov11 = mx.expand_dims(mx.expand_dims(t_inv_cov[:, 1, 1], -1), -1)
        
        row0 = dx * icov00 + dy * icov10
        row1 = dx * icov01 + dy * icov11
        pow_val = -0.5 * (dx * row0 + dy * row1)
        
        alpha = mx.exp(pow_val) * mx.expand_dims(mx.expand_dims(t_ops, -1), -1)
        mask = mx.expand_dims(mx.expand_dims(local_mask, -1), -1) & (pow_val > -10.0)
        alpha_eff = mx.where(mask, mx.minimum(0.99, alpha), 0.0)
        
        # Transmission
        one_minus_alpha = 1.0 - alpha_eff
        T = mx.concatenate([
            mx.ones((1, tile_size, tile_size)), 
            mx.cumprod(one_minus_alpha[:-1], axis=0)
        ], axis=0)
        
        # Blending
        weights = alpha_eff * T
        accum_color = mx.sum(mx.expand_dims(weights, -1) * t_cols[:, None, None, :], axis=0)
        
        final_T = T[-1] * one_minus_alpha[-1]
        final_tile = accum_color + mx.expand_dims(final_T, -1) * background
        return final_tile

    starts = tile_boundaries[:-1]
    ends = tile_boundaries[1:]
    all_tiles = mx.vmap(rasterize_single_tile)(starts, ends, gy_tiles, gx_tiles)
    
    # Reassemble
    output_grid = all_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    output_image = output_grid.transpose(0, 2, 1, 3, 4).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    
    return output_image[:H, :W, :]
