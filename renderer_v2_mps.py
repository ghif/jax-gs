"""
TODO:
Fix this code to run faster!
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from gaussians import Gaussians, get_covariance_3d

# Constants
TILE_SIZE = 16
MAX_TILES_PER_SPLAT = 64 # Enough for splats covering ~8x8 blocks (128x128 pixels)
BLOCK_SIZE = 256  # Max Gaussians per tile to process (front-to-back)

class Camera(NamedTuple):
    W: int      # Image width
    H: int      # Image height
    fx: float   # Focal length x
    fy: float   # Focal length y
    cx: float   # Principal point x
    cy: float   # Principal point y
    W2C: jnp.ndarray  # World-to-Camera matrix (4, 4)
    full_proj: jnp.ndarray  # Full projection matrix (4, 4)

def project_gaussians(gaussians: Gaussians, camera: Camera):
    """
    Project 3D Gaussians to 2D splats and compute bounding boxes.
    """
    means3D = gaussians.means
    scales = gaussians.scales
    quats = gaussians.quaternions
    
    # 1. Transform means to camera space
    # Force float32 for constants
    means3D_homo = jnp.concatenate([means3D, jnp.ones((means3D.shape[0], 1), dtype=jnp.float32)], axis=-1)
    means_cam = (means3D_homo @ camera.W2C.T)[:, :3]
    
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    
    # 2. Filter out Gaussians behind the camera/frustum or too close
    valid_mask = z > jnp.float32(0.01)  # Near plane
    
    # 3. Get 3D covariance
    cov3D = get_covariance_3d(scales, quats)
    
    # Project to 2D
    # Jacobian approximation
    # Cast intrinsics to float32 to avoid float64 promotion
    fx = jnp.float32(camera.fx)
    fy = jnp.float32(camera.fy)
    cx = jnp.float32(camera.cx)
    cy = jnp.float32(camera.cy)
    
    J = jnp.zeros((means3D.shape[0], 2, 3), dtype=jnp.float32)
    J = J.at[:, 0, 0].set(fx / z)
    J = J.at[:, 0, 2].set(-fx * x / (z**2))
    J = J.at[:, 1, 1].set(fy / z)
    J = J.at[:, 1, 2].set(-fy * y / (z**2))
    
    W_rot = camera.W2C[:3, :3]
    
    # cov2D = J @ W @ cov3D @ W.T @ J.T
    def project_single_cov(c3d, j_mat):
        return j_mat @ W_rot @ c3d @ W_rot.T @ j_mat.T
    
    cov2D = jax.vmap(project_single_cov)(cov3D, J)
    
    # Add smoothing/eps
    cov2D = cov2D.at[:, 0, 0].add(jnp.float32(0.3))
    cov2D = cov2D.at[:, 1, 1].add(jnp.float32(0.3))
    
    # 5. Compute means 2D
    means2D = jnp.stack([
        fx * x / z + cx,
        fy * y / z + cy
    ], axis=-1)
    
    # 6. Compute Radii (Bounding Box)
    # Using 3 sigma of the major eigenvalue
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    
    # Eigenvalues of 2x2 matrix: tr/2 +/- sqrt((tr/2)^2 - det)
    mid = trace / jnp.float32(2.0)
    term = jnp.sqrt(jnp.maximum(mid**2 - det, jnp.float32(0.0)))
    lambda1 = mid + term
    lambda2 = mid - term
    max_eigen = jnp.maximum(lambda1, lambda2)
    
    radii = jnp.ceil(jnp.float32(3.0) * jnp.sqrt(max_eigen))
    
    return means2D, cov2D, radii, valid_mask, z

def get_tile_interactions(means2D, radii, valid_mask, depths, H, W, tile_size):
    """
    Generate the list of (tile_id, gaussian_id, depth) interactions.
    Sorted by tile_id, then depth.

    Args:
        means2D: (N, 2) array of 2D means
        radii: (N,) array of radii
        valid_mask: (N,) boolean array of valid points
        depths: (N,) array of depths
        H: int, height of image
        W: int, width of image
        tile_size: int, size of tile
    Returns:
        interactions: (N, 3) array of (tile_id, gaussian_id, depth)
    """
    num_points = means2D.shape[0]
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    
    # Bounding box in tile coordinates
    min_x = jnp.clip((means2D[:, 0] - radii), jnp.float32(0.0), jnp.float32(W - 1))
    max_x = jnp.clip((means2D[:, 0] + radii), jnp.float32(0.0), jnp.float32(W - 1))
    min_y = jnp.clip((means2D[:, 1] - radii), jnp.float32(0.0), jnp.float32(H - 1))
    max_y = jnp.clip((means2D[:, 1] + radii), jnp.float32(0.0), jnp.float32(H - 1))
    
    tile_min_x = (min_x // tile_size).astype(jnp.int32)
    tile_max_x = (max_x // tile_size).astype(jnp.int32)
    tile_min_y = (min_y // tile_size).astype(jnp.int32)
    tile_max_y = (max_y // tile_size).astype(jnp.int32)
    
    # Check bounds
    # Reject if min > max (invalid) or masked out
    valid_mask = valid_mask & (tile_max_x >= tile_min_x) & (tile_max_y >= tile_min_y)
    
    # Generate potential interactions
    # We create a fixed pattern of offsets to cover the max area
    # Max tiles = MAX_TILES_PER_SPLAT.
    
    def get_gaussian_tiles(idx, t_min_x, t_max_x, t_min_y, t_max_y, is_valid):
        # Create grid 
        xs = jnp.arange(0, 8) # Assume max spread 8 tiles
        ys = jnp.arange(0, 8)
        grid_y, grid_x = jnp.meshgrid(ys, xs, indexing='ij') # (8, 8)
        
        # Absolute tile coords
        abs_x = t_min_x + grid_x
        abs_y = t_min_y + grid_y
        
        # Valid mask for this tile in the grid
        in_range = (abs_x <= t_max_x) & (abs_y <= t_max_y) & is_valid
        
        # Compute tile ID: y * dims + x
        tile_ids = abs_y * num_tiles_x + abs_x
        
        # Set invalid to -1
        tile_ids = jnp.where(in_range, tile_ids, -1)
        
        return tile_ids.flatten()

    # Vmap over all gaussians
    all_tile_ids = jax.vmap(get_gaussian_tiles)(
        jnp.arange(num_points), 
        tile_min_x, tile_max_x, 
        tile_min_y, tile_max_y, 
        valid_mask
    ) # (N, 64)
    
    # Corresponding gaussian IDs
    all_gaussian_ids = jnp.broadcast_to(jnp.arange(num_points)[:, None], all_tile_ids.shape)
    all_depths = jnp.broadcast_to(depths[:, None], all_tile_ids.shape)
    
    # Flatten
    flat_tile_ids = all_tile_ids.reshape(-1)
    flat_gaussian_ids = all_gaussian_ids.reshape(-1)
    flat_depths = all_depths.reshape(-1)
    
    # Filter out -1 tile ids
    valid_interactions = flat_tile_ids != -1
    
    # Replace invalid keys with high value for sorting to end
    sort_tile_ids = jnp.where(valid_interactions, flat_tile_ids, jnp.iinfo(jnp.int32).max)
    
    # MPS Optimization: Single-pass argsort with int32 packed key
    # Key = (tile_id << 13) | depth_quantized
    # 13 bits for depth = 8192 levels. Sufficient for local sorting within tile.
    # 31 - 13 = 18 bits for tile_id = 262,144 tiles.
    # For 16x16 tiles, that supports image size ~8000x8000.
    
    DEPTH_BITS = 13
    
    # Quantize depth to [0, 2^DEPTH_BITS - 1]
    # We assume reasonable depth range, or use monotonic mapping?
    # Simple linear mapping: 
    # To preserve order, we just need monotonic transformation.
    # However, to pack into bits, we need integer.
    # We can interpret float bytes as int for sorting, but we need to shift.
    # Safer: normalize min/max or use view-space z directly?
    # Since we sort per tile, global ordering matters for tile_id, 
    # but for depth only local ordering matters.
    # Actually, we just need (tile_a < tile_b) OR (tile_a == tile_b AND depth_a < depth_b).
    #
    # Quantizing valid depths (e.g. 0.1 to 100.0) to u13:
    # We can clamp to a range and scale.
    # Or bitcast? bitcast preserves order for positive floats.
    # bitcast float32 -> int32.
    # Shift right to keep top 13 bits?
    # top bits of float are sign, exponent, mantissa.
    # For positive floats, larger float = larger int rep.
    # keeping top 13 bits retains exponent and top mantissa.
    # That is effectively "logarithmic quantization". Very good for depth!
    
    depth_i32_full = jax.lax.bitcast_convert_type(flat_depths, jnp.int32)
    # We want top 13 bits. But we don't want negative (sign bit 31). z > 0.
    # Shift down: 31 (sign) + 8 (exp) + 23 (mantissa).
    # We want to keep 13 bits of "significance".
    # int32 is 31 bits magnitude.
    # We can't just << tile_id because tile_id needs high bits.
    # Key layout: [TileID: 18 bits] [Depth: 13 bits]
    
    # Depth reduction:
    # Take top 13 bits of the Int32 representation (excluding sign bit, or assume positive).
    # depth is always positive (> 0.01).
    # Mask out sign bit just in case: & 0x7FFFFFFF (not needed for logic shift if we cast to uint? No uint32 on MPS int32 sort?)
    # MPS argsort works on int32.
    # depth_i32 >> (31 - 13) = depth_i32 >> 18.
    
    depth_quant = depth_i32_full >> (31 - DEPTH_BITS)
    
    # Mask to ensure within bits (though shift handles it)
    # depth_quant is now 13 bits (mostly exponent).
    
    # Pack:
    # tile_id is int32.
    key = (sort_tile_ids << DEPTH_BITS) | depth_quant
    
    # Sort
    sort_indices = jnp.argsort(key)
    
    sorted_tile_ids = sort_tile_ids[sort_indices]
    sorted_gaussian_ids = flat_gaussian_ids[sort_indices]
    
    return sorted_tile_ids, sorted_gaussian_ids, valid_interactions.sum()

def render_tiles(means2D, cov2D, opacities, colors, depths, 
                 sorted_tile_ids, sorted_gaussian_ids, 
                 H, W, tile_size=TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)):
    
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    # Precompute inverse covariances
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = jnp.maximum(det, jnp.float32(1e-6))
    inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)
    
    def rasterize_single_tile(tile_idx):
        # 1. Find range in sorted list
        start_idx = jnp.searchsorted(sorted_tile_ids, tile_idx)
        end_idx = jnp.searchsorted(sorted_tile_ids, tile_idx + 1)
        
        # Get count
        count = end_idx - start_idx
        
        indices = jax.lax.dynamic_slice(sorted_gaussian_ids, (start_idx,), (BLOCK_SIZE,))
        
        # Mask out if we went past the end or if count < BLOCK_SIZE
        local_mask = jnp.arange(BLOCK_SIZE) < count
        
        # 2. Fetch Gaussian Data
        # Ensure indices are valid (clamp to 0 if masked out, but mask handles it)
        safe_indices = jnp.where(local_mask, indices, 0)
        
        t_means = means2D[safe_indices]
        t_inv_cov = inv_cov2D[safe_indices]
        t_ops = opacities[safe_indices]
        t_cols = colors[safe_indices] # (N, 3)
        
        # Tile coords
        ty = tile_idx // num_tiles_x
        tx = tile_idx % num_tiles_x
        
        pix_min_x = tx * tile_size
        pix_min_y = ty * tile_size
        
        # 3. Iterate over pixels (16x16)
        py, px = jnp.mgrid[0:tile_size, 0:tile_size]
        pixel_x = pix_min_x + px
        pixel_y = pix_min_y + py
        
        pixel_coords = jnp.stack([pixel_x, pixel_y], axis=-1) # (16, 16, 2)
        
        # Flatten pixels for scanning
        pixel_coords_flat = pixel_coords.reshape(-1, 2)
        
        # Check bounds (for edge tiles)
        pixel_valid = (pixel_coords_flat[:, 0] < W) & (pixel_coords_flat[:, 1] < H)
        
        # Inner loop: blend for one pixel
        def blend_pixel(p_coord, p_valid):
            
            def scan_fn(carry, i):
                accum_color, T = carry
                
                # Retrieve validation
                # Ensure T comparison is float32
                is_valid = local_mask[i] & (T > jnp.float32(1e-4))
                
                # Data
                mu = t_means[i]
                icov = t_inv_cov[i]
                op = t_ops[i, 0]
                col = t_cols[i]
                
                # Gaussian evaluation
                d = p_coord - mu
                power = -0.5 * (d[0] * (d[0] * icov[0, 0] + d[1] * icov[1, 0]) + 
                               d[1] * (d[0] * icov[0, 1] + d[1] * icov[1, 1]))
                
                # Ensure 0.99 is float32
                alpha = jnp.minimum(jnp.float32(0.99), jnp.exp(power) * jax.nn.sigmoid(op))
                
                # Visibility test
                # Ensure -10.0 is float32
                visible = (power > jnp.float32(-10.0)) & is_valid
                
                alpha = jnp.where(visible, alpha, 0.0)
                
                # Blend
                weight = alpha * T
                new_color = accum_color + weight * col
                new_T = T * (jnp.float32(1.0) - alpha)
                
                return (new_color, new_T), None
            
            # Initial carry args must be float32
            init_color = jnp.zeros(3, dtype=jnp.float32)
            init_T = jnp.float32(1.0)
            
            (final_color, final_T), _ = jax.lax.scan(scan_fn, (init_color, init_T), jnp.arange(BLOCK_SIZE))
            
            # Background blending
            final_color = final_color + final_T * background
            
            return jnp.where(p_valid, final_color, 0.0)

        # Vectorize over 256 pixels
        tile_colors = jax.vmap(blend_pixel)(pixel_coords_flat, pixel_valid)
        
        return tile_colors.reshape(tile_size, tile_size, 3)

    # Vmap over tiles
    all_tiles = jax.vmap(rasterize_single_tile)(jnp.arange(num_tiles))
    
    # Assemble image
    output_grid = all_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    output_image = output_grid.swapaxes(1, 2).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    
    # Crop to actual Size
    return output_image[:H, :W, :]

def render_v2_mps(gaussians: Gaussians, camera: Camera, background=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)):
    """
    Main entry point for tile-based rendering (MPS optimized).
    """
    return render_camera_v2_mps(gaussians, camera.W2C, camera.fx, camera.fy, camera.cx, camera.cy, camera.W, camera.H, background)

def render_camera_v2_mps(gaussians: Gaussians, W2C, fx, fy, cx, cy, W, H, background=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)):
    """
    Decomposed rendering function friendly for JIT with static_argnums for W and H.
    """
    # Reconstruct temporary camera tuple for internal usage
    camera = Camera(W, H, fx, fy, cx, cy, W2C, jnp.eye(4, dtype=jnp.float32))
    
    # 1. Project
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera)
    
    # 2. Precompute Colors (SH)
    # Simple DC term
    # Ensure constants are float32
    colors = gaussians.sh_coeffs[:, 0, :] * jnp.float32(0.28209479177387814) + jnp.float32(0.5)
    colors = jnp.clip(colors, jnp.float32(0.0), jnp.float32(1.0))
    
    # 3. Sort Interactions
    # Using the fix inside get_tile_interactions (we modified it in this file)
    sorted_tile_ids, sorted_gaussian_ids, _ = get_tile_interactions(
        means2D, radii, valid_mask, depths, H, W, TILE_SIZE
    )
    
    # 4. Rasterize
    image = render_tiles(
        means2D, cov2D, gaussians.opacities, colors, depths,
        sorted_tile_ids, sorted_gaussian_ids,
        H, W, TILE_SIZE, background
    )
    
    return image
