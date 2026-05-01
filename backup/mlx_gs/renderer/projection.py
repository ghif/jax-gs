import mlx.core as mx
from mlx_gs.core.gaussians import get_covariance_3d

def project_gaussians(params, camera_dict):
    """
    Project 3D Gaussians to 2D splats in MLX.
    
    Args:
        params: dict of Gaussians (means, scales, quaternions)
        camera_dict: dict of camera params (W, H, fx, fy, cx, cy, W2C)

    Returns:
        means2D: (N, 2) array of 2D means
        cov2D: (N, 2, 2) array of 2D covariances
        radii: (N,) array of radii
        valid_mask: (N,) boolean array of valid Gaussians
        z: (N,) array of depths
    """
    means3D = params["means"]
    scales = params["scales"]
    quats = params["quaternions"]
    
    W = camera_dict["W"]
    H = camera_dict["H"]
    fx = camera_dict["fx"]
    fy = camera_dict["fy"]
    cx = camera_dict["cx"]
    cy = camera_dict["cy"]
    W2C = camera_dict["W2C"]
    
    # 1. Transform means
    means3D_homo = mx.concatenate([means3D, mx.ones((means3D.shape[0], 1))], axis=-1)
    means_cam = (means3D_homo @ mx.transpose(W2C))[:, :3]
    
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    
    # 2. Filter 
    valid_mask = z > 0.01
    
    # 3. Covariance
    cov3D = get_covariance_3d(scales, quats)
    
    # 4. Project to 2D
    # Jacobian
    # Differentiable Jacobian construction
    J = mx.stack([
        mx.stack([fx / z, mx.zeros_like(z), -fx * x / (z**2)], axis=-1),
        mx.stack([mx.zeros_like(z), fy / z, -fy * y / (z**2)], axis=-1)
    ], axis=-2)
    
    W_rot = W2C[:3, :3]
    
    # Project cov2D: J @ W_rot @ cov3D @ W_rot.T @ J.T
    W_rot_T = mx.transpose(W_rot)
    T1 = J @ W_rot      # (N, 2, 3)
    T2 = T1 @ cov3D     # (N, 2, 3)
    T3 = T2 @ W_rot_T   # (N, 2, 3)
    cov2D_raw = T3 @ mx.transpose(J, (0, 2, 1)) # (N, 2, 2)
    
    # Add a small bias (low pass filter) differentiably
    # We construct the 2x2 matrix from components
    c00 = cov2D_raw[:, 0, 0] + 0.3
    c01 = cov2D_raw[:, 0, 1]
    c10 = cov2D_raw[:, 1, 0]
    c11 = cov2D_raw[:, 1, 1] + 0.3
    
    cov2D = mx.stack([
        mx.stack([c00, c01], axis=-1),
        mx.stack([c10, c11], axis=-1)
    ], axis=-2)
    
    # 5. Means 2D
    means2D = mx.stack([
        fx * x / z + cx,
        fy * y / z + cy
    ], axis=-1)
    
    # 6. Radii for tile interaction
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    mid = trace / 2.0
    term = mx.sqrt(mx.maximum(mid**2 - det, 1e-7))
    lambda1 = mid + term
    max_eigen = lambda1 
    radii = mx.ceil(3.0 * mx.sqrt(max_eigen))
    
    return means2D, cov2D, radii, valid_mask, z
