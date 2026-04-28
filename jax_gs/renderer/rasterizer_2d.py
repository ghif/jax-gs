import jax
import jax.numpy as jnp
from typing import Tuple
from jax_gs.renderer.rasterizer import get_tile_interactions, TILE_SIZE, BLOCK_SIZE

def render_tiles_2d(means2D, cov2D, opacities, colors, depths, normals, sorted_tile_ids, sorted_gaussian_ids, 
                    H, W, tile_size: int = TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0])):
    """
    Render the tiles for 2DGS. Outputs color, depth, and normal maps.
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

    py, px = jnp.mgrid[0:tile_size, 0:tile_size]
    tile_pixel_x = px.astype(jnp.float32)
    tile_pixel_y = py.astype(jnp.float32)
    
    def rasterize_single_tile(tile_idx):
        start_idx = tile_boundaries[tile_idx]
        end_idx = tile_boundaries[tile_idx + 1]
        count = end_idx - start_idx
        
        gather_indices = jnp.clip(start_idx + jnp.arange(BLOCK_SIZE), 0, sorted_gaussian_ids.shape[0] - 1)
        indices = jnp.take(sorted_gaussian_ids, gather_indices)
        local_mask = (start_idx + jnp.arange(BLOCK_SIZE)) < (start_idx + count)
        
        t_means = means2D[indices]
        t_inv_cov = inv_cov2D[indices]
        t_ops = sig_opacities[indices]
        t_cols = colors[indices]
        t_depths = depths[indices]
        t_normals = normals[indices]
        
        ty = tile_idx // num_tiles_x
        tx = tile_idx % num_tiles_x
        pix_min_x = (tx * tile_size).astype(jnp.float32)
        pix_min_y = (ty * tile_size).astype(jnp.float32)
        
        grid_x = pix_min_x + tile_pixel_x
        grid_y = pix_min_y + tile_pixel_y
        pixel_coords = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
        pixel_valid = (pixel_coords[:, 0] < W) & (pixel_coords[:, 1] < H)

        t_ic00 = t_inv_cov[:, 0, 0]
        t_ic01_2 = 2.0 * t_inv_cov[:, 0, 1]
        t_ic11 = t_inv_cov[:, 1, 1]
        t_op_vec = t_ops[:, 0]
        
        def process_tile():
            def blend_pixel(p_coord, p_valid):
                def scan_fn(carry, i):
                    accum_color, accum_depth, accum_depth_sq, accum_normal, T = carry
                    is_active = local_mask[i] & (T > 1e-4)
                    
                    mu = t_means[i]
                    dx = p_coord[0] - mu[0]
                    dy = p_coord[1] - mu[1]
                    
                    power = -0.5 * (dx * dx * t_ic00[i] + dx * dy * t_ic01_2[i] + dy * dy * t_ic11[i])
                    
                    alpha = jnp.exp(power) * t_op_vec[i]
                    alpha = jnp.where((power > -10.0) & is_active, jnp.minimum(0.99, alpha), 0.0)
                    
                    weight = alpha * T
                    new_T = T * (1.0 - alpha)
                    new_color = accum_color + weight * t_cols[i]
                    new_depth = accum_depth + weight * t_depths[i]
                    new_depth_sq = accum_depth_sq + weight * (t_depths[i] ** 2)
                    new_normal = accum_normal + weight * t_normals[i]
                    
                    return (new_color, new_depth, new_depth_sq, new_normal, new_T), None
                
                (final_color, final_depth, final_depth_sq, final_normal, final_T), _ = jax.lax.scan(
                    scan_fn, (jnp.zeros(3), 0.0, 0.0, jnp.zeros(3), 1.0), jnp.arange(BLOCK_SIZE)
                )
                
                final_color = final_color + final_T * background
                # Return all components as a single array for vmap efficiency
                res = jnp.concatenate([
                    final_color,
                    jnp.array([final_depth, final_depth_sq]),
                    final_normal,
                    jnp.array([1.0 - final_T])
                ])
                return jnp.where(p_valid, res, jnp.zeros(9))

            tile_data = jax.vmap(blend_pixel)(pixel_coords, pixel_valid)
            return tile_data.reshape(tile_size, tile_size, 9)
            
        def empty_tile():
            bg = jnp.zeros(9)
            bg = bg.at[0:3].set(background)
            return jnp.broadcast_to(bg, (tile_size, tile_size, 9))

        return jax.lax.cond(count > 0, process_tile, empty_tile)

    # Parallel rasterization
    all_tiles = jax.vmap(rasterize_single_tile)(jnp.arange(num_tiles))
    
    # Reshape and crop
    def untile(data):
        # data: (num_tiles, tile_size, tile_size, C)
        grid = data.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, -1)
        grid = grid.swapaxes(1, 2) # (num_tiles_y, tile_size, num_tiles_x, tile_size, C)
        full = grid.reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, -1)
        return full[:H, :W, :]

    full_data = untile(all_tiles)
    
    return (full_data[:, :, 0:3], 
            full_data[:, :, 3:4], 
            full_data[:, :, 4:5], 
            full_data[:, :, 5:8], 
            full_data[:, :, 8:9])
