import jax
import jax.numpy as jnp
from typing import Tuple

# Standard Constants
TILE_SIZE = 16  # Each tile is 16x16 pixels
BLOCK_SIZE = 512  # Increased from 192 to 512 to reduce blocky artifacts on datasets with high depth complexity

def get_tile_interactions(means2D, radii, valid_mask, depths, H, W, tile_size: int = TILE_SIZE):
    """
    Determines which Gaussians overlap which 16x16 tiles and sorts them by tile ID and depth.
    
    This is the 'Culling' and 'Sorting' phase of Gaussian Splatting. Instead of custom CUDA
    kernels, we use JAX's vectorized operations and bit-packed sorting for efficiency.

    Args:
        means2D: [N, 2] Projected 2D centers of Gaussians.
        radii: [N] Screen-space radii for bounding box calculation.
        valid_mask: [N] Boolean mask for active Gaussians.
        depths: [N] Depth from camera for front-to-back sorting.
        H, W: Image dimensions.
        tile_size: Size of square tiles.
    Returns:
        sorted_tile_ids: [M] Tile IDs for each interaction, sorted.
        sorted_gaussian_ids: [M] Gaussian IDs for each interaction, sorted by tile then depth.
        valid_interactions: Scalar count of non-padded interactions.
    """
    num_points = means2D.shape[0]
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size

    # Calculate bounding boxes in pixel coordinates
    min_x = jnp.clip((means2D[:, 0] - radii), 0, W - 1)
    max_x = jnp.clip((means2D[:, 0] + radii), 0, W - 1)
    min_y = jnp.clip((means2D[:, 1] - radii), 0, H - 1)
    max_y = jnp.clip((means2D[:, 1] + radii), 0, H - 1)

    # Convert pixel bounding boxes to tile coordinate ranges
    tile_min_x = (min_x // tile_size).astype(jnp.int32)
    tile_max_x = (max_x // tile_size).astype(jnp.int32)
    tile_min_y = (min_y // tile_size).astype(jnp.int32)
    tile_max_y = (max_y // tile_size).astype(jnp.int32)

    # Filter points completely outside the screen
    on_screen = (means2D[:, 0] + radii > 0) & (means2D[:, 0] - radii < W) & \
                (means2D[:, 1] + radii > 0) & (means2D[:, 1] - radii < H)

    valid_mask = valid_mask & on_screen & (tile_max_x >= tile_min_x) & (tile_max_y >= tile_min_y)

    # Pre-calculate relative tile offsets for broadcasting.
    # OFFSET_SIZE defines the max size of a Gaussian in tiles (16 tiles = 256 pixels).
    # We use a static meshgrid to assign Gaussians to multiple tiles simultaneously.
    OFFSET_SIZE = 16
    off_y, off_x = jnp.meshgrid(jnp.arange(OFFSET_SIZE), jnp.arange(OFFSET_SIZE), indexing='ij')
    off_x = off_x.flatten()
    off_y = off_y.flatten()

    # Calculate absolute tile coordinates for every potential Gaussian-tile pair
    abs_x = tile_min_x[:, None] + off_x[None, :]
    abs_y = tile_min_y[:, None] + off_y[None, :]

    # Check if the calculated tile is within the Gaussian's actual bounding box
    in_range = (abs_x <= tile_max_x[:, None]) & (abs_y <= tile_max_y[:, None]) & valid_mask[:, None]

    # Map (y, x) tile coordinates to a single 1D Tile ID
    all_tile_ids = abs_y * num_tiles_x + abs_x
    all_tile_ids = jnp.where(in_range, all_tile_ids, -1) # -1 indicates no interaction

    # Replicate Gaussian IDs and Depths for every potential tile interaction
    all_gaussian_ids = jnp.broadcast_to(jnp.arange(num_points)[:, None], all_tile_ids.shape)
    all_depths = jnp.broadcast_to(depths[:, None], all_tile_ids.shape)

    flat_tile_ids = all_tile_ids.reshape(-1)
    flat_gaussian_ids = all_gaussian_ids.reshape(-1)
    flat_depths = all_depths.reshape(-1)

    valid_interactions = flat_tile_ids != -1

    # ROBUST PACK-SORT: We need to sort by Tile ID (primary) and then Depth (secondary).
    # Instead of two sorts, we pack both into a single 32-bit integer:
    # [TileID (18 bits)][Depth (13 bits)]
    DEPTH_BITS = 13
    num_tiles_total = num_tiles_x * num_tiles_y

    # 1. Prepare Primary Key (Tile ID)
    # Invalid interactions are moved to the end by using num_tiles_total as a sentinel.
    sort_tile_ids = jnp.where(valid_interactions, flat_tile_ids, num_tiles_total)

    # 2. Prepare Secondary Key (Depth)
    # We quantize the float32 depth into 13 bits to fit in the int32 pack.
    depth_i32_full = jax.lax.bitcast_convert_type(flat_depths, jnp.int32)
    depth_quant = depth_i32_full >> (31 - DEPTH_BITS)

    # 3. Pack and Sort
    # The left-shift puts TileID in the most significant bits, ensuring it's the primary sort key.
    key = (sort_tile_ids << DEPTH_BITS) | depth_quant

    # lax.sort_key_val is a highly optimized primitive for sorting key-value pairs.
    sorted_keys, sorted_gaussian_ids = jax.lax.sort_key_val(key, flat_gaussian_ids)

    # Extract the original Tile ID from the sorted key
    sorted_tile_ids = sorted_keys >> DEPTH_BITS

    # Padding: Ensure the output has a minimum size for consistent JIT compilation.
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
    Renders the image by processing each tile and alpha-blending the overlapping Gaussians.
    
    This function implements the standard front-to-back alpha blending:
    Color = sum(alpha_i * T_i * Color_i) + T_final * Background
    where T_i is the accumulated transmittance.

    Args:
        means2D: [N, 2] Gaussian centers.
        cov2D: [N, 2, 2] 2D Covariance matrices.
        opacities: [N, 1] Raw opacities (pre-sigmoid).
        colors: [N, 3] Gaussian colors.
        sorted_tile_ids, sorted_gaussian_ids: Output from get_tile_interactions.
        H, W: Image dimensions.
        background: RGB background color.
    Returns:
        image: [H, W, 3] The rendered image.
    """
    
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    # Invert 2D covariances for the Gaussian exponent: power = -0.5 * (x-mu)^T * Sigma^-1 * (x-mu)
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = jnp.maximum(det, 1e-6)
    
    inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)

    # Convert raw opacities to [0, 1] range
    sig_opacities = jax.nn.sigmoid(opacities)
    
    # Use searchsorted to find the start/end index of each tile's Gaussians in the sorted list.
    tile_indices = jnp.arange(num_tiles + 1)
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, tile_indices)

    # Pre-calculate local pixel offsets within a tile
    py, px = jnp.mgrid[0:tile_size, 0:tile_size]
    tile_pixel_x = px.astype(jnp.float32) + 0.5
    tile_pixel_y = py.astype(jnp.float32) + 0.5
    
    def rasterize_single_tile(tile_idx):
        """Processes a single 16x16 tile."""
        start_idx = tile_boundaries[tile_idx]
        end_idx = tile_boundaries[tile_idx + 1]
        count = end_idx - start_idx
        
        # Gather the Gaussians for this specific tile.
        # We process Gaussians in chunks of BLOCK_SIZE for JIT-friendliness.
        gather_indices = jnp.clip(start_idx + jnp.arange(BLOCK_SIZE), 0, sorted_gaussian_ids.shape[0] - 1)
        indices = jnp.take(sorted_gaussian_ids, gather_indices)
        local_mask = (start_idx + jnp.arange(BLOCK_SIZE)) < (start_idx + count)
        
        t_means = means2D[indices]
        t_inv_cov = inv_cov2D[indices]
        t_ops = sig_opacities[indices]
        t_cols = colors[indices]
        
        # Calculate global pixel coordinates for this tile
        ty = tile_idx // num_tiles_x
        tx = tile_idx % num_tiles_x
        pix_min_x = (tx * tile_size).astype(jnp.float32)
        pix_min_y = (ty * tile_size).astype(jnp.float32)
        
        grid_x = pix_min_x + tile_pixel_x
        grid_y = pix_min_y + tile_pixel_y
        pixel_coords = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
        pixel_valid = (pixel_coords[:, 0] < W) & (pixel_coords[:, 1] < H)

        # Pre-extract covariance components for optimized math in the inner loop
        t_ic00 = t_inv_cov[:, 0, 0]
        t_ic01_2 = 2.0 * t_inv_cov[:, 0, 1]
        t_ic11 = t_inv_cov[:, 1, 1]
        t_op_vec = t_ops[:, 0]
        
        def process_tile():
            def blend_pixel(p_coord, p_valid):
                """Alpha-blends all Gaussians for a single pixel."""
                def scan_fn(carry, i):
                    accum_color, T = carry
                    # Early exit if transmittance T is nearly zero (pixel is opaque)
                    is_active = local_mask[i] & (T > 1e-4)
                    
                    mu = t_means[i]
                    dx = p_coord[0] - mu[0]
                    dy = p_coord[1] - mu[1]
                    
                    # Compute the Gaussian influence at this pixel
                    power = -0.5 * (dx * dx * t_ic00[i] + dx * dy * t_ic01_2[i] + dy * dy * t_ic11[i])
                    
                    alpha = jnp.exp(power) * t_op_vec[i]
                    # Clamp alpha and mask inactive/far-away Gaussians
                    alpha = jnp.where((power > -10.0) & is_active, jnp.minimum(0.99, alpha), 0.0)
                    
                    # Standard Front-to-Back Blending:
                    # T_next = T_current * (1 - alpha)
                    # Color_next = Color_current + T_current * alpha * Color_i
                    new_T = T * (1.0 - alpha)
                    new_color = accum_color + (alpha * T) * t_cols[i]
                    
                    return (new_color, new_T), None
                
                # lax.scan performs the loop efficiently on the device
                (final_color, final_T), _ = jax.lax.scan(scan_fn, (jnp.zeros(3), 1.0), jnp.arange(BLOCK_SIZE))
                # Add background color weighted by remaining transmittance
                final_color = final_color + final_T * background
                return jnp.where(p_valid, final_color, 0.0)

            # Parallelize over pixels in the tile
            tile_image = jax.vmap(blend_pixel)(pixel_coords, pixel_valid)
            return tile_image.reshape(tile_size, tile_size, 3)
            
        def empty_tile():
            """Optimization for tiles with no Gaussians."""
            return jnp.broadcast_to(background, (tile_size, tile_size, 3))

        return jax.lax.cond(count > 0, process_tile, empty_tile)

    # Parallelize over all tiles in the image
    all_tiles = jax.vmap(rasterize_single_tile)(jnp.arange(num_tiles))
    
    # Reshape tiles back into the full image
    output_grid = all_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    output_image = output_grid.swapaxes(1, 2).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    
    # Crop to original image dimensions (removing padding)
    return output_image[:H, :W, :]