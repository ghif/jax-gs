import jax
import jax.numpy as jnp
from typing import Tuple

# Standard Constants
TILE_SIZE = 16
BLOCK_SIZE = 256  

def get_tile_interactions(means2D, radii, valid_mask, depths, H, W, tile_size: int = TILE_SIZE):
    """
    Generate tile interactions.
    Uses bit-packed int32 sort for broad backend compatibility.

    Args:
        means2D: 2D means of the projected splats
        radii: Radii of the projected splats
        valid_mask: Valid mask for the projected splats
        depths: Depth of the projected splats
        H: Image height
        W: Image width
        tile_size: Tile size
    Returns:
        sorted_tile_ids: Sorted tile IDs
        sorted_gaussian_ids: Sorted Gaussian IDs
        valid_interactions: Number of valid interactions
    """
    num_points = means2D.shape[0]
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    
    min_x = jnp.clip((means2D[:, 0] - radii), 0, W - 1)
    max_x = jnp.clip((means2D[:, 0] + radii), 0, W - 1)
    min_y = jnp.clip((means2D[:, 1] - radii), 0, H - 1)
    max_y = jnp.clip((means2D[:, 1] + radii), 0, H - 1)
    
    tile_min_x = (min_x // tile_size).astype(jnp.int32)
    tile_max_x = (max_x // tile_size).astype(jnp.int32)
    tile_min_y = (min_y // tile_size).astype(jnp.int32)
    tile_max_y = (max_y // tile_size).astype(jnp.int32)
    
    # Filter points completely outside image
    on_screen = (means2D[:, 0] + radii > 0) & (means2D[:, 0] - radii < W) & \
                (means2D[:, 1] + radii > 0) & (means2D[:, 1] - radii < H)
    
    valid_mask = valid_mask & on_screen & (tile_max_x >= tile_min_x) & (tile_max_y >= tile_min_y)
    
    def get_gaussian_tiles(idx, t_min_x, t_max_x, t_min_y, t_max_y, is_valid):
        """
        Get the tiles that a Gaussian intersects with.

        Args:
            idx: Gaussian index
            t_min_x: Minimum tile x coordinate
            t_max_x: Maximum tile x coordinate
            t_min_y: Minimum tile y coordinate
            t_max_y: Maximum tile y coordinate
            is_valid: Valid mask for the Gaussian
        Returns:
            tile_ids: Tile IDs that the Gaussian intersects with
        """
        xs = jnp.arange(0, 8) 
        ys = jnp.arange(0, 8)
        grid_y, grid_x = jnp.meshgrid(ys, xs, indexing='ij') 
        
        abs_x = t_min_x + grid_x
        abs_y = t_min_y + grid_y
        
        in_range = (abs_x <= t_max_x) & (abs_y <= t_max_y) & is_valid
        
        tile_ids = abs_y * num_tiles_x + abs_x
        tile_ids = jnp.where(in_range, tile_ids, -1)
        
        return tile_ids.flatten()

    all_tile_ids = jax.vmap(get_gaussian_tiles)(
        jnp.arange(num_points), 
        tile_min_x, tile_max_x, 
        tile_min_y, tile_max_y, 
        valid_mask
    )
    
    all_gaussian_ids = jnp.broadcast_to(jnp.arange(num_points)[:, None], all_tile_ids.shape)
    all_depths = jnp.broadcast_to(depths[:, None], all_tile_ids.shape)
    
    flat_tile_ids = all_tile_ids.reshape(-1)
    flat_gaussian_ids = all_gaussian_ids.reshape(-1)
    flat_depths = all_depths.reshape(-1)
    
    valid_interactions = flat_tile_ids != -1
    
    # Robust Pack-Sort: [TileID: 18 bits] [Depth: 13 bits]
    # We use bit-packing to sort by (tile_id, depth) using a single integer argsort.
    DEPTH_BITS = 13
    num_tiles_total = num_tiles_x * num_tiles_y
    
    # 1. Prepare Primary Key (Tile ID)
    # Use num_tiles_total as sentinel (guaranteed > any valid tile_id)
    # This keeps values small enough for int32 (assuming < 200k tiles)
    sort_tile_ids = jnp.where(valid_interactions, flat_tile_ids, num_tiles_total)
    
    # 2. Prepare Secondary Key (Depth)
    depth_i32_full = jax.lax.bitcast_convert_type(flat_depths, jnp.int32)
    depth_quant = depth_i32_full >> (31 - DEPTH_BITS)
    
    # 3. Pack (all in int32)
    key = (sort_tile_ids << DEPTH_BITS) | depth_quant
    sort_indices = jnp.argsort(key)
    
    sorted_tile_ids = sort_tile_ids[sort_indices]
    sorted_gaussian_ids = flat_gaussian_ids[sort_indices]
    
    # Ensure at least BLOCK_SIZE for dynamic_slice in rasterizer
    # Use python max() to keep it a concrete integer for jnp.full/at
    total_interactions = sorted_tile_ids.shape[0]
    padded_size = max(total_interactions, BLOCK_SIZE)
    
    pad_tile_ids = jnp.full((padded_size,), num_tiles_total, dtype=jnp.int32)
    pad_gaussian_ids = jnp.zeros((padded_size,), dtype=jnp.int32)
    
    sorted_tile_ids = pad_tile_ids.at[:total_interactions].set(sorted_tile_ids)
    sorted_gaussian_ids = pad_gaussian_ids.at[:total_interactions].set(sorted_gaussian_ids)
    
    return sorted_tile_ids, sorted_gaussian_ids, valid_interactions.sum()

def render_tiles(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                 H, W, tile_size: int = TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0])):
    """
    Render the tiles.

    Args:
        means2D: 2D means of the projected splats
        cov2D: 2D covariance of the projected splats
        opacities: Opacities of the projected splats
        colors: Colors of the projected splats
        sorted_tile_ids: Sorted tile IDs
        sorted_gaussian_ids: Sorted Gaussian IDs
        H: Image height
        W: Image width
        tile_size: Tile size
        background: Background color
    Returns:
        image: Rendered image
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
    
    def rasterize_single_tile(tile_idx):
        """
        Rasterize a single tile.

        Args:
            tile_idx: Tile index
        Returns:
            image: Rendered image
        """
        start_idx = jnp.searchsorted(sorted_tile_ids, tile_idx)
        end_idx = jnp.searchsorted(sorted_tile_ids, tile_idx + 1)
        count = end_idx - start_idx
        
        indices = jax.lax.dynamic_slice(sorted_gaussian_ids, (start_idx,), (BLOCK_SIZE,))
        local_mask = jnp.arange(BLOCK_SIZE) < count
        safe_indices = jnp.where(local_mask, indices, 0)
        
        t_means = means2D[safe_indices]
        t_inv_cov = inv_cov2D[safe_indices]
        t_ops = opacities[safe_indices]
        t_cols = colors[safe_indices]
        
        ty = tile_idx // num_tiles_x
        tx = tile_idx % num_tiles_x
        pix_min_x = tx * tile_size
        pix_min_y = ty * tile_size
        
        py, px = jnp.mgrid[0:tile_size, 0:tile_size]
        pixel_x = pix_min_x + px
        pixel_y = pix_min_y + py
        pixel_coords = jnp.stack([pixel_x, pixel_y], axis=-1).reshape(-1, 2)
        pixel_valid = (pixel_coords[:, 0] < W) & (pixel_coords[:, 1] < H)
        
        def process_tile():
            def blend_pixel(p_coord, p_valid):
                def scan_fn(carry, i):
                    accum_color, T = carry
                    is_valid = local_mask[i] & (T > 1e-4)
                    
                    mu = t_means[i]
                    icov = t_inv_cov[i]
                    op = t_ops[i, 0]
                    col = t_cols[i]
                    
                    d = p_coord - mu
                    power = -0.5 * (d[0] * (d[0] * icov[0, 0] + d[1] * icov[1, 0]) + 
                                   d[1] * (d[0] * icov[0, 1] + d[1] * icov[1, 1]))
                    
                    alpha = jnp.minimum(0.99, jnp.exp(power) * jax.nn.sigmoid(op))
                    visible = (power > -10.0) & is_valid
                    alpha = jnp.where(visible, alpha, 0.0)
                    
                    weight = alpha * T
                    new_color = accum_color + weight * col
                    new_T = T * (1.0 - alpha)
                    
                    return (new_color, new_T), None
                
                (final_color, final_T), _ = jax.lax.scan(scan_fn, (jnp.zeros(3), 1.0), jnp.arange(BLOCK_SIZE))
                final_color = final_color + final_T * background
                return jnp.where(p_valid, final_color, 0.0)

            return jax.vmap(blend_pixel)(pixel_coords, pixel_valid)
            
        def empty_tile():
            return jnp.tile(background, (tile_size * tile_size, 1))

        tile_colors = jax.lax.cond(count > 0, process_tile, empty_tile)
        return tile_colors.reshape(tile_size, tile_size, 3)

    # Rasterize all tiles in parallel
    all_tiles = jax.vmap(rasterize_single_tile)(jnp.arange(num_tiles))
    
    output_grid = all_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    output_image = output_grid.swapaxes(1, 2).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    
    return output_image[:H, :W, :]
