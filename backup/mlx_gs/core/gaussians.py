import mlx.core as mx

class GaussiansMLX:
    def __init__(self, means, scales, quaternions, opacities, sh_coeffs):
        self.means = means
        self.scales = scales
        self.quaternions = quaternions
        self.opacities = opacities
        self.sh_coeffs = sh_coeffs

def get_covariance_3d(scales: mx.array, quaternions: mx.array):
    """
    Computes 3D covariance matrix from scales and quaternions in MLX.
    """
    # Normalize quaternions with epsilon for gradient stability
    q = quaternions / (mx.linalg.norm(quaternions, axis=-1, keepdims=True) + 1e-7)
    
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    # MLX stack and reshape for rotation matrix
    # R: (N, 3, 3)
    R = mx.stack([
        mx.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*r*z, 2*x*z + 2*r*y], axis=-1),
        mx.stack([2*x*y + 2*r*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*r*x], axis=-1),
        mx.stack([2*x*z - 2*r*y, 2*y*z + 2*r*x, 1 - 2*x**2 - 2*y**2], axis=-1)
    ], axis=-2)
    
    # Scaling matrix
    s = mx.exp(scales)
    # Differentiable diagonal matrix construction
    S = mx.stack([
        mx.stack([s[:, 0], mx.zeros_like(s[:, 0]), mx.zeros_like(s[:, 0])], axis=-1),
        mx.stack([mx.zeros_like(s[:, 0]), s[:, 1], mx.zeros_like(s[:, 0])], axis=-1),
        mx.stack([mx.zeros_like(s[:, 0]), mx.zeros_like(s[:, 0]), s[:, 2]], axis=-1)
    ], axis=-2)
    
    # M = R S
    M = R @ S
    
    # Σ = M M^T
    Sigma = M @ mx.transpose(M, (0, 2, 1))
    
    return Sigma

def init_gaussians_from_pcd(points, colors):
    """
    Initialize Gaussians from a point cloud in MLX.
    """
    num_points = points.shape[0]
    
    means = points
    scales = mx.full((num_points, 3), -3.0) 
    quaternions = mx.tile(mx.array([1.0, 0.0, 0.0, 0.0]), (num_points, 1))
    opacities = mx.full((num_points, 1), 0.0) 
    
    sh_dc = (colors - 0.5) / 0.28209479177387814
    sh_coeffs = mx.zeros((num_points, 16, 3))
    sh_coeffs[:, 0, :] = sh_dc
    
    return {
        "means": means,
        "scales": scales,
        "quaternions": quaternions,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs
    }
