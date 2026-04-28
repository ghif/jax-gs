import jax
import jax.numpy as jnp
import chex

@chex.dataclass
class Gaussians2D:
    means: jnp.ndarray  # (N, 3)
    scales: jnp.ndarray  # (N, 2) - Only two scales for 2D disks
    quaternions: jnp.ndarray  # (N, 4)
    opacities: jnp.ndarray  # (N, 1)
    sh_coeffs: jnp.ndarray  # (N, K, 3) where K is num SH coefficients

def get_tangent_vectors(quaternions: jnp.ndarray):
    """
    Computes tangent vectors u, v and normal vector n from quaternions.
    The local disk is in the XY plane, and the normal is along Z.
    
    Args:
        quaternions: (N, 4)
    Returns:
        u: (N, 3) - Tangent vector 1 (transformed local X)
        v: (N, 3) - Tangent vector 2 (transformed local Y)
        n: (N, 3) - Normal vector (transformed local Z)
    """
    # Normalize quaternions
    q = quaternions / jnp.linalg.norm(quaternions, axis=-1, keepdims=True)
    
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    # Rotation matrix R columns
    # R = [u | v | n]
    u = jnp.stack([
        1 - 2*y**2 - 2*z**2,
        2*x*y + 2*r*z,
        2*x*z - 2*r*y
    ], axis=-1)
    
    v = jnp.stack([
        2*x*y - 2*r*z,
        1 - 2*x**2 - 2*z**2,
        2*y*z + 2*r*x
    ], axis=-1)
    
    n = jnp.stack([
        2*x*z + 2*r*y,
        2*y*z - 2*r*x,
        1 - 2*x**2 - 2*y**2
    ], axis=-1)
    
    return u, v, n

def get_covariance_2d_world(scales: jnp.ndarray, quaternions: jnp.ndarray):
    """
    Computes the world-space representation of the 2D Gaussian.
    This isn't a 3D covariance matrix (which would be rank-deficient),
    but rather the tangent vectors scaled by the 2D scales.
    
    Args:
        scales: (N, 2)
        quaternions: (N, 4)
    Returns:
        M: (N, 3, 2) - Matrix [s1*u | s2*v]
    """
    u, v, _ = get_tangent_vectors(quaternions)
    s = jnp.exp(scales)
    
    M = jnp.stack([u * s[:, 0:1], v * s[:, 1:2]], axis=-1)
    return M

def init_gaussians_2d_from_pcd(points: jnp.ndarray, colors: jnp.ndarray):
    """
    Initialize 2D Gaussians from a point cloud.

    Args:
        points: (N, 3)
        colors: (N, 3) in [0, 1]
    Returns:
        gaussians: Gaussians2D dataclass
    """
    num_points = points.shape[0]
    
    # Position: mean of the point cloud
    means = points
    
    # Scales: log of the distance to the nearest neighbors
    # Initialized to a small value (approx 0.05m)
    scales = jnp.full((num_points, 2), -3.0) 
    
    # Rotations: identity quaternions [1, 0, 0, 0]
    quaternions = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (num_points, 1))
    
    # Opacities: inverse sigmoid of 0.5 = 0.0
    opacities = jnp.full((num_points, 1), 0.0) 
    
    # SH Coefficients (DC term only)
    sh_dc = (colors - 0.5) / 0.28209479177387814
    sh_coeffs = jnp.zeros((num_points, 16, 3)) # Degree 3 SH -> 16 coefficients
    sh_coeffs = sh_coeffs.at[:, 0, :].set(sh_dc)
    
    return Gaussians2D(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        sh_coeffs=sh_coeffs
    )
