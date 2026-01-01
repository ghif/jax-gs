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
    means3D_homo = jnp.concatenate([means3D, jnp.ones((means3D.shape[0], 1))], axis=-1)
    means_cam = (means3D_homo @ camera.W2C.T)[:, :3]
    
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    
    # 2. Filter out Gaussians behind the camera/frustum or too close
    valid_mask = z > 0.01  # Near plane
    
    # 3. Get 3D covariance
    cov3D = get_covariance_3d(scales, quats)
    
    # 4. Project to 2D
    # Jacobian approximation
    J = jnp.zeros((means3D.shape[0], 2, 3))
    J = J.at[:, 0, 0].set(camera.fx / z)
    J = J.at[:, 0, 2].set(-camera.fx * x / (z**2))
    J = J.at[:, 1, 1].set(camera.fy / z)
    J = J.at[:, 1, 2].set(-camera.fy * y / (z**2))
    
    W_rot = camera.W2C[:3, :3]
    
    # cov2D = J @ W @ cov3D @ W.T @ J.T
    def project_single_cov(c3d, j_mat):
        return j_mat @ W_rot @ c3d @ W_rot.T @ j_mat.T
    
    cov2D = jax.vmap(project_single_cov)(cov3D, J)
    
    # Add smoothing/eps
    cov2D = cov2D.at[:, 0, 0].add(0.3)
    cov2D = cov2D.at[:, 1, 1].add(0.3)
    
    # 5. Compute means 2D
    means2D = jnp.stack([
        camera.fx * x / z + camera.cx,
        camera.fy * y / z + camera.cy
    ], axis=-1)
    
    # 6. Compute Radii (Bounding Box)
    # Using 3 sigma of the major eigenvalue
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    
    # Eigenvalues of 2x2 matrix: tr/2 +/- sqrt((tr/2)^2 - det)
    mid = trace / 2.0
    term = jnp.sqrt(jnp.maximum(mid**2 - det, 0.0))
    lambda1 = mid + term
    lambda2 = mid - term
    max_eigen = jnp.maximum(lambda1, lambda2)
    
    radii = jnp.ceil(3.0 * jnp.sqrt(max_eigen))
    
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
    min_x = jnp.clip((means2D[:, 0] - radii), 0, W - 1)
    max_x = jnp.clip((means2D[:, 0] + radii), 0, W - 1)
    min_y = jnp.clip((means2D[:, 1] - radii), 0, H - 1)
    max_y = jnp.clip((means2D[:, 1] + radii), 0, H - 1)
    
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
    # We map a flat index 0..MAX-1 to relative dx, dy offsets?
    # No, simple approach:
    # return a list of tile IDs.
    
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
    # We limit to 8x8=64 tiles max coverage per gaussian. if > 8x8, it's clamped.
    # This is a constraint.
    
    # Note: If MAX_TILES_PER_SPLAT is 64, that is 8x8.
    
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
    
    # Sort
    # Primary key: tile_id, Secondary: depth
    # lexsort sorts by last key first? JAX lexsort: keys are (k1, k2, ...). 
    # jnp.lexsort((secondary, primary))
    
    sort_indices = jnp.lexsort((flat_depths, sort_tile_ids))
    
    sorted_tile_ids = sort_tile_ids[sort_indices]
    sorted_gaussian_ids = flat_gaussian_ids[sort_indices]
    
    return sorted_tile_ids, sorted_gaussian_ids, valid_interactions.sum()

def render_tiles(means2D, cov2D, opacities, colors, depths, 
                 sorted_tile_ids, sorted_gaussian_ids, 
                 H, W, tile_size=TILE_SIZE, background=jnp.array([0.0, 0.0, 0.0])):
    
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    # Precompute inverse covariances
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = jnp.maximum(det, 1e-6)
    inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)
    
    def rasterize_single_tile(tile_idx):
        # 1. Find range in sorted list
        # We need to find where tile_idx starts and ends.
        # Since sorted_tile_ids is sorted, we can use searchsorted.
        start_idx = jnp.searchsorted(sorted_tile_ids, tile_idx)
        end_idx = jnp.searchsorted(sorted_tile_ids, tile_idx + 1)
        
        # Get count
        count = end_idx - start_idx
        
        # Slice the gaussian IDs
        # We must use dynamic_slice or just simple slicing logic with padding
        # JAX requires static slice size for vmap?
        # Actually searchsorted returns dynamic scalar.
        # We can use lax.dynamic_slice, but the shape must be inferable?
        # No, dynamic_slice output shape is static (size argument).
        
        indices = jax.lax.dynamic_slice(sorted_gaussian_ids, (start_idx,), (BLOCK_SIZE,))
        
        # Mask out if we went past the end or if count < BLOCK_SIZE
        # We can create a mask based on arange(BLOCK_SIZE) < count
        local_mask = jnp.arange(BLOCK_SIZE) < count
        
        # 2. Fetch Gaussian Data
        # If index is out of bounds (due to dynamic slice reading junk), gather will handle specific behavior?
        # indices could contain values from the next tile if we just slice.
        # So we must safeguard the indices.
        
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
        # We can vmap this or use mgrid
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
                
                # Check if we should stop (T < 1e-4) -> Early exit optimization not easy in JAX scan
                # But we can just gate updates.
                
                # Retrieve validation
                is_valid = local_mask[i] & (T > 1e-4) # & p_valid
                
                # Data
                mu = t_means[i]
                icov = t_inv_cov[i]
                op = t_ops[i, 0]
                col = t_cols[i]
                
                # Gaussian evaluation
                d = p_coord - mu
                power = -0.5 * (d[0] * (d[0] * icov[0, 0] + d[1] * icov[1, 0]) + 
                               d[1] * (d[0] * icov[0, 1] + d[1] * icov[1, 1]))
                
                alpha = jnp.minimum(0.99, jnp.exp(power) * jax.nn.sigmoid(op))
                
                # Visibility test
                visible = (power > -10.0) & is_valid
                
                alpha = jnp.where(visible, alpha, 0.0)
                
                # Blend
                weight = alpha * T
                new_color = accum_color + weight * col
                new_T = T * (1.0 - alpha)
                
                return (new_color, new_T), None
            
            (final_color, final_T), _ = jax.lax.scan(scan_fn, (jnp.zeros(3), 1.0), jnp.arange(BLOCK_SIZE))
            
            # Background blending
            final_color = final_color + final_T * background
            
            return jnp.where(p_valid, final_color, 0.0)

        # Vectorize over 256 pixels
        tile_colors = jax.vmap(blend_pixel)(pixel_coords_flat, pixel_valid)
        
        return tile_colors.reshape(tile_size, tile_size, 3)

    # Vmap over tiles
    all_tiles = jax.vmap(rasterize_single_tile)(jnp.arange(num_tiles))
    
    # Assemble image
    # all_tiles: (num_tiles, 16, 16, 3)
    # We need to reshape to (H, W, 3)
    # This requires careful reshaping.
    # Dimensions: (Ny, Nx, 16, 16, 3)
    # Transpose to (Ny, 16, Nx, 16, 3) -> (H, W, 3)
    
    output_grid = all_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    output_image = output_grid.swapaxes(1, 2).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    
    # Crop to actual Size
    return output_image[:H, :W, :]



def render_v2(gaussians: Gaussians, camera: Camera, background=jnp.array([0.0, 0.0, 0.0])):
    """
    Main entry point for tile-based rendering.
    NOTE: When JIT-compiling, ensure that `camera.H` and `camera.W` are static.
    If using standard JIT, better to use the decomposed function below.
    """
    return render_camera_v2(gaussians, camera.W2C, camera.fx, camera.fy, camera.cx, camera.cy, camera.W, camera.H, background)

def render_camera_v2(gaussians: Gaussians, W2C, fx, fy, cx, cy, W, H, background=jnp.array([0.0, 0.0, 0.0])):
    """
    Decomposed rendering function friendly for JIT with static_argnums for W and H.
    """
    # Reconstruct temporary camera tuple for internal usage
    camera = Camera(W, H, fx, fy, cx, cy, W2C, jnp.eye(4))
    
    # 1. Project
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera)
    
    # 2. Precompute Colors (SH)
    # Simple DC term
    colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
    colors = jnp.clip(colors, 0.0, 1.0)
    
    # 3. Sort Interactions
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
