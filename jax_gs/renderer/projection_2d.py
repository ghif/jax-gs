import jax
import jax.numpy as jnp
from jax_gs.core.gaussians_2d import Gaussians2D, get_tangent_vectors
from jax_gs.core.camera import Camera

def project_gaussians_2d(gaussians: Gaussians2D, camera: Camera):
    """
    Project 2D Gaussians to 2D splats using perspective-correct projection.

    Args:
        gaussians: Gaussians2D dataclass
        camera: Camera dataclass 
    Returns:
        means2D: 2D means of the projected splats
        cov2D: 2D covariance of the projected splats
        radii: Radii of the projected splats
        valid_mask: Valid mask for the projected splats
        z: Depth of the projected splats
        normals: Normals in camera space (for regularization)
    """
    means3D = gaussians.means
    scales = gaussians.scales
    quats = gaussians.quaternions
    
    # 1. Transform means to camera space
    means3D_homo = jnp.concatenate([means3D, jnp.ones((means3D.shape[0], 1))], axis=-1)
    means_cam = (means3D_homo @ camera.W2C.T)[:, :3]
    
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    
    # 2. Filter (Near plane clipping)
    valid_mask = z > 0.01
    
    # 3. Tangent vectors and Normals
    u_world, v_world, n_world = get_tangent_vectors(quats)
    
    # Transform to camera space
    W_rot = camera.W2C[:3, :3]
    u_cam = u_world @ W_rot.T
    v_cam = v_world @ W_rot.T
    normals = n_world @ W_rot.T # Normals in camera space
    
    # 4. Jacobian of perspective transformation
    inv_z = 1.0 / jnp.maximum(z, 0.01)
    inv_z2 = inv_z**2
    
    # J = [fx/z, 0, -fx*x/z^2]
    #     [0, fy/z, -fy*y/z^2]
    J = jnp.stack([
        jnp.stack([camera.fx * inv_z, jnp.zeros_like(z), -camera.fx * x * inv_z2], axis=-1),
        jnp.stack([jnp.zeros_like(z), camera.fy * inv_z, -camera.fy * y * inv_z2], axis=-1)
    ], axis=-2)
    
    # 5. Project tangent vectors to 2D
    # M = J @ [u_cam | v_cam] (N, 2, 2)
    M = J @ jnp.stack([u_cam, v_cam], axis=-1)
    
    # Apply scales
    s = jnp.exp(scales)
    # Scaled columns: M[:, :, 0] * s[:, 0], M[:, :, 1] * s[:, 1]
    M_scaled = M * s[:, None, :]
    
    # 6. Compute 2D covariance
    # cov2D = M_scaled @ M_scaled^T
    cov2D = M_scaled @ M_scaled.transpose(0, 2, 1)
    
    # Add low-pass filter (anti-aliasing)
    eye2D = jnp.eye(2)[None, :, :]
    cov2D = cov2D + 0.3 * eye2D
    
    # 7. Means 2D
    means2D = jnp.stack([
        camera.fx * x / z + camera.cx,
        camera.fy * y / z + camera.cy
    ], axis=-1)
    
    # 8. Radii for tile interaction
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    mid = trace / 2.0
    term = jnp.sqrt(jnp.maximum(mid**2 - det, 0.0))
    lambda1 = mid + term
    max_eigen = lambda1 
    radii = jnp.ceil(3.0 * jnp.sqrt(max_eigen))
    
    return means2D, cov2D, radii, valid_mask, z, normals
