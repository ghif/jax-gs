import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from gaussians import Gaussians, get_covariance_3d

# Constants
# For NVIDIA L4 (Ada Lovelace), we can likely afford slightly larger block sizes if needed,
# but 256 is a reasonable standard for tile-based rasterization.
TILE_SIZE = 16
BLOCK_SIZE = 256  

class Camera(NamedTuple):
    W: int      
    H: int      
    fx: float   
    fy: float   
    cx: float   
    cy: float   
    W2C: jnp.ndarray  
    full_proj: jnp.ndarray  

def project_gaussians(gaussians: Gaussians, camera: Camera):
    """
    Project 3D Gaussians to 2D splats.
    Optimized for CUDA: Standard JAX ops work well here.
    """
    means3D = gaussians.means
    scales = gaussians.scales
    quats = gaussians.quaternions
    
    # 1. Transform means
    means3D_homo = jnp.concatenate([means3D, jnp.ones((means3D.shape[0], 1))], axis=-1)
    means_cam = (means3D_homo @ camera.W2C.T)[:, :3]
    
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    
    # 2. Filter 
    valid_mask = z > 0.01
    
    # 3. Covariance
    cov3D = get_covariance_3d(scales, quats)
    
    # 4. Project to 2D
    J = jnp.zeros((means3D.shape[0], 2, 3))
    J = J.at[:, 0, 0].set(camera.fx / z)
    J = J.at[:, 0, 2].set(-camera.fx * x / (z**2))
    J = J.at[:, 1, 1].set(camera.fy / z)
    J = J.at[:, 1, 2].set(-camera.fy * y / (z**2))
    
    W_rot = camera.W2C[:3, :3]
    
    def project_single_cov(c3d, j_mat):
        return j_mat @ W_rot @ c3d @ W_rot.T @ j_mat.T
    
    cov2D = jax.vmap(project_single_cov)(cov3D, J)
    cov2D = cov2D.at[:, 0, 0].add(0.3)
    cov2D = cov2D.at[:, 1, 1].add(0.3)
    
    # 5. Means 2D
    means2D = jnp.stack([
        camera.fx * x / z + camera.cx,
        camera.fy * y / z + camera.cy
    ], axis=-1)
    
    # 6. Radii
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    mid = trace / 2.0
    term = jnp.sqrt(jnp.maximum(mid**2 - det, 0.0))
    lambda1 = mid + term
    max_eigen = lambda1 
    radii = jnp.ceil(3.0 * jnp.sqrt(max_eigen))
    
    return means2D, cov2D, radii, valid_mask, z

def get_tile_interactions(means2D, radii, valid_mask, depths, H, W, tile_size):
    """
    Generate tile interactions.
    CUDA Optimized: We can use jnp.argsort heavily.
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
    
    valid_mask = valid_mask & (tile_max_x >= tile_min_x) & (tile_max_y >= tile_min_y)
    
    def get_gaussian_tiles(idx, t_min_x, t_max_x, t_min_y, t_max_y, is_valid):
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
    
    # Sort
    # On CUDA (and CPU), jnp.lexsort is efficient and preferred.
    # We sort by tile_id (primary) and depth (secondary).
    
    # Filter valid first to reduce sort size? 
    # Or strict sort with invalid at end.
    
    sort_tile_ids = jnp.where(valid_interactions, flat_tile_ids, jnp.iinfo(jnp.int32).max)
    
    # Lexsort: sorts by last key first.
    # keys: (flat_depths, sort_tile_ids) -> sorts by tile_ids, then depths.
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
    
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = jnp.maximum(det, 1e-6)
    inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)
    
    def rasterize_single_tile(tile_idx):
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
        
        pixel_coords = jnp.stack([pixel_x, pixel_y], axis=-1)
        pixel_coords_flat = pixel_coords.reshape(-1, 2)
        pixel_valid = (pixel_coords_flat[:, 0] < W) & (pixel_coords_flat[:, 1] < H)
        
        # Optimization: Early exit if tile is empty
        # This is beneficial on GPU too to avoid launching threads for empty tiles
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

            return jax.vmap(blend_pixel)(pixel_coords_flat, pixel_valid)
            
        def empty_tile():
            # (256, 3)
            return jnp.tile(background, (tile_size * tile_size, 1))

        tile_colors = jax.lax.cond(count > 0, process_tile, empty_tile)
        return tile_colors.reshape(tile_size, tile_size, 3)

    all_tiles = jax.vmap(rasterize_single_tile)(jnp.arange(num_tiles))
    
    output_grid = all_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    output_image = output_grid.swapaxes(1, 2).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    
    return output_image[:H, :W, :]

def render_v2_gpu(gaussians: Gaussians, camera: Camera, background=jnp.array([0.0, 0.0, 0.0])):
    return render_camera_v2_gpu(gaussians, camera.W2C, camera.fx, camera.fy, camera.cx, camera.cy, camera.W, camera.H, background)

def render_camera_v2_gpu(gaussians: Gaussians, W2C, fx, fy, cx, cy, W, H, background=jnp.array([0.0, 0.0, 0.0])):
    camera = Camera(W, H, fx, fy, cx, cy, W2C, jnp.eye(4))
    
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera)
    
    colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
    colors = jnp.clip(colors, 0.0, 1.0)
    
    sorted_tile_ids, sorted_gaussian_ids, _ = get_tile_interactions(
        means2D, radii, valid_mask, depths, H, W, TILE_SIZE
    )
    
    image = render_tiles(
        means2D, cov2D, gaussians.opacities, colors, depths,
        sorted_tile_ids, sorted_gaussian_ids,
        H, W, TILE_SIZE, background
    )
    
    return image
