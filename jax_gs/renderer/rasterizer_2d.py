import jax
import jax.numpy as jnp
from typing import Tuple
from jax_gs.renderer.rasterizer import get_tile_interactions, TILE_SIZE, BLOCK_SIZE

def render_tiles_2d(means2D, cov2D, opacities, colors, depths, normals, sorted_tile_ids, sorted_gaussian_ids, 
                    H, W, tile_size: int = TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0])):
    """
    Render the tiles for 2DGS. Outputs color, depth, and normal maps.

    Args:
        means2D: 2D means of the projected splats
        cov2D: 2D covariance of the projected splats
        opacities: Opacities of the projected splats
        colors: Colors of the projected splats
        depths: Depths of the projected splats
        normals: Normals of the projected splats (in camera space)
        sorted_tile_ids: Sorted tile IDs
        sorted_gaussian_ids: Sorted Gaussian IDs
        H: Image height
        W: Image width
        tile_size: Tile size
        background: Background color
    Returns:
        image: Rendered image (H, W, 3)
        depth_map: Rendered depth map (H, W, 1)
        depth_sq_map: Rendered depth squared map (H, W, 1) - for distortion loss
        normal_map: Rendered normal map (H, W, 3)
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
    # Pre-calculate sigmoid opacities
    sig_opacities = jax.nn.sigmoid(opacities)
    
    # Pre-calculate tile boundaries
    tile_indices = jnp.arange(num_tiles + 1)
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, tile_indices)

    # Pre-calculate pixel grid for a single tile
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
        safe_indices = indices
        
        t_means = means2D[safe_indices]
        t_inv_cov = inv_cov2D[safe_indices]
        t_ops = sig_opacities[safe_indices]
        t_cols = colors[safe_indices]
        t_depths = depths[safe_indices]
        t_normals = normals[safe_indices]
        
        ty = tile_idx // num_tiles_x
        tx = tile_idx % num_tiles_x
        pix_min_x = (tx * tile_size).astype(jnp.float32)
        pix_min_y = (ty * tile_size).astype(jnp.float32)
        
        grid_x = pix_min_x + tile_pixel_x
        grid_y = pix_min_y + tile_pixel_y
        pixel_coords = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
        pixel_valid = (pixel_coords[:, 0] < W) & (pixel_coords[:, 1] < H)

        # Pre-extract covariance components for expanded math
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
                # We don't add background to depth or normals typically, or we add a far value
                return jnp.where(p_valid, 
                                 jnp.concatenate([final_color, 
                                                  jnp.array([final_depth]), 
                                                  jnp.array([final_depth_sq]), 
                                                  final_normal]), 
                                 jnp.zeros(8))

            tile_data = jax.vmap(blend_pixel)(pixel_coords, pixel_valid)
            return tile_data.reshape(tile_size, tile_size, 8)
            
        def empty_tile():
            # color (3), depth (1), depth_sq (1), normal (3)
            bg_data = jnp.concatenate([background, jnp.array([0.0, 0.0]), jnp.zeros(3)])
            return jnp.broadcast_to(bg_data, (tile_size, tile_size, 8))

        tile_data = jax.lax.cond(count > 0, process_tile, empty_tile)
        return tile_data

    # Rasterize all tiles in parallel
    all_tiles = jax.vmap(rasterize_single_tile)(jnp.arange(num_tiles))
    
    output_grid = all_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 8)
    output_data = output_grid.swapaxes(1, 2).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 8)
    output_data = output_data[:H, :W, :]
    
    image = output_data[:, :, 0:3]
    depth_map = output_data[:, :, 3:4]
    depth_sq_map = output_data[:, :, 4:5]
    normal_map = output_data[:, :, 5:8]
    
    return image, depth_map, depth_sq_map, normal_map
