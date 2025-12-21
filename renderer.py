import jax
import jax.numpy as jnp
from typing import NamedTuple
from gaussians import Gaussians, get_covariance_3d

class Camera(NamedTuple):
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float
    W2C: jnp.ndarray  # (4, 4)
    full_proj: jnp.ndarray  # (4, 4)

def project_gaussians(gaussians: Gaussians, camera: Camera):
    """
    Project 3D Gaussians to 2D splats.
    """
    means3D = gaussians.means
    scales = gaussians.scales
    quats = gaussians.quaternions
    
    # 1. Transform means to camera space
    means3D_homo = jnp.concatenate([means3D, jnp.ones((means3D.shape[0], 1))], axis=-1)
    means_cam = (means3D_homo @ camera.W2C.T)[:, :3]
    
    # 2. Get 3D covariance
    cov3D = get_covariance_3d(scales, quats)
    
    # 3. Project to 2D (EWA Approximation)
    # J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
    # cov2D = J W cov3D W^T J^T
    
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    
    # Filter out Gaussians behind the camera
    valid_mask = z > 0.01
    
    # Jacobian of the perspective projection
    J = jnp.zeros((means3D.shape[0], 2, 3))
    J = J.at[:, 0, 0].set(camera.fx / z)
    J = J.at[:, 0, 2].set(-camera.fx * x / (z**2))
    J = J.at[:, 1, 1].set(camera.fy / z)
    J = J.at[:, 1, 2].set(-camera.fy * y / (z**2))
    
    # W is the rotation part of W2C
    W = camera.W2C[:3, :3]
    
    # cov2D = J @ W @ cov3D @ W.T @ J.T
    # Using vmap for batching
    def project_single_cov(c3d, j_mat):
        return j_mat @ W @ c3d @ W.T @ j_mat.T
    
    cov2D = jax.vmap(project_single_cov)(cov3D, J)
    
    # Add a small smoothing term to the 2D covariance (anti-aliasing)
    cov2D = cov2D.at[:, 0, 0].add(0.3)
    cov2D = cov2D.at[:, 1, 1].add(0.3)
    
    # 2D means (image coordinates)
    means2D = jnp.stack([
        camera.fx * x / z + camera.cx,
        camera.fy * y / z + camera.cy
    ], axis=-1)
    
    return means2D, cov2D, valid_mask, z

def render(gaussians: Gaussians, camera: Camera):
    """
    Differentiable rendering of Gaussians.
    """
    # 1. Project Gaussians
    means2D, cov2D, valid_mask, depths = project_gaussians(gaussians, camera)
    
    # 2. Filter invalid (behind camera)
    depths = jnp.where(valid_mask, depths, 1e10)
    
    # 3. Sort by depth (front to back)
    indices = jnp.argsort(depths)
    means2D = means2D[indices]
    cov2D = cov2D[indices]
    opacities = gaussians.opacities[indices]
    
    # Compute colors from SH (DC term only for now)
    # SH_DC factor = 0.28209479177387814
    colors = gaussians.sh_coeffs[indices, 0, :] * 0.28209479177387814 + 0.5
    colors = jnp.clip(colors, 0.0, 1.0)
    
    # 4. Rasterize (Per-pixel blending)
    H, W = camera.H, camera.W
    pixel_y, pixel_x = jnp.mgrid[0:H, 0:W]
    pixel_coords = jnp.stack([pixel_x, pixel_y], axis=-1).reshape(-1, 2) # (H*W, 2)
    
    # Pre-compute inverse 2D covariances
    dets = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
    # Handle zero determinants by adding small epsilon
    dets = jnp.maximum(dets, 1e-6)
    
    inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / dets, -cov2D[:, 0, 1] / dets], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / dets, cov2D[:, 0, 0] / dets], axis=-1)
    ], axis=-2)
    
    # We only want to process Gaussians that have some influence.
    # For simplicity, we take the top N Gaussians.
    max_gaussians = 500 # Adjust based on memory
    top_indices = jnp.arange(min(means2D.shape[0], max_gaussians))
    
    m2d = means2D[top_indices]
    icov2d = inv_cov2D[top_indices]
    cols = colors[top_indices]
    ops = opacities[top_indices]
    
    def blend_pixel(coords):
        # Scan over Gaussians
        def step(carry, i):
            accum_color, T = carry
            
            diff = coords - m2d[i]
            
            # power = -0.5 * diff.T @ inv_cov2D @ diff
            p = -0.5 * (diff[0] * (diff[0] * icov2d[i, 0, 0] + diff[1] * icov2d[i, 1, 0]) + 
                       diff[1] * (diff[0] * icov2d[i, 0, 1] + diff[1] * icov2d[i, 1, 1]))
            
            # Limit influence to a certain radius for performance/stability
            # exp(-3) is approx 0.05, so we can cut off there
            alpha = jnp.exp(p) * jax.nn.sigmoid(ops[i, 0])
            alpha = jnp.where(p < -10.0, 0.0, alpha) # Cutoff
            
            # Alpha blending: C = sum(color_i * alpha_i * T_i)
            new_color = accum_color + cols[i] * alpha * T
            new_T = T * (1.0 - alpha)
            
            return (new_color, new_T), None

        (final_color, _), _ = jax.lax.scan(step, (jnp.zeros(3), 1.0), top_indices)
        return final_color

    # Vmap over all pixels
    image = jax.vmap(blend_pixel)(pixel_coords)
    image = image.reshape(H, W, 3)
    
    return image
