import jax.numpy as jnp
from gaussians import init_gaussians_from_pcd, get_covariance_3d

def test_init_gaussians_from_pcd():
    num_points = 10
    points = jnp.zeros((num_points, 3))
    colors = jnp.ones((num_points, 3)) # White colors
    
    gaussians = init_gaussians_from_pcd(points, colors)
    
    assert gaussians.means.shape == (num_points, 3)
    assert gaussians.scales.shape == (num_points, 3)
    assert gaussians.quaternions.shape == (num_points, 4)
    assert gaussians.opacities.shape == (num_points, 1)
    assert gaussians.sh_coeffs.shape == (num_points, 16, 3)
    
    # Check SH DC term initialization
    # DC term for 1.0 (white) should be (1.0 - 0.5) / 0.28209...
    expected_dc = (1.0 - 0.5) / 0.28209479177387814
    assert jnp.allclose(gaussians.sh_coeffs[:, 0, :], expected_dc)

def test_get_covariance_3d():
    # Case 1: Identity rotation, Unit scale (log scale = 0)
    scales = jnp.zeros((1, 3)) 
    quaternions = jnp.array([[1.0, 0.0, 0.0, 0.0]]) # Identity
    
    cov = get_covariance_3d(scales, quaternions)
    
    # Expected covariance is Identity * exp(0)^2 = Identity
    assert cov.shape == (1, 3, 3)
    assert jnp.allclose(cov[0], jnp.eye(3))
    
    # Case 2: Scaling
    scales = jnp.log(jnp.array([[2.0, 1.0, 0.5]]))
    cov = get_covariance_3d(scales, quaternions)
    
    expected_cov = jnp.diag(jnp.array([4.0, 1.0, 0.25]))
    assert jnp.allclose(cov[0], expected_cov)
    
    # Case 3: Rotation (90 deg around Z)
    # q = [cos(45), 0, 0, sin(45)] = [0.707, 0, 0, 0.707]
    scales = jnp.log(jnp.array([[2.0, 1.0, 1.0]])) # Scale X by 2
    angle = jnp.pi / 2
    quaternions = jnp.array([[jnp.cos(angle/2), 0.0, 0.0, jnp.sin(angle/2)]])
    
    cov = get_covariance_3d(scales, quaternions)
    
    # After 90 deg rot around Z, X-axis becomes Y-axis.
    # So the scaling of 2 on X should now appear on Y in the covariance matrix?
    # R * S * S^T * R^T
    # S = diag(2, 1, 1)
    # R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # M = R S = [[0, -1, 0], [2, 0, 0], [0, 0, 1]]
    # Sigma = M M^T = [[1, 0, 0], [0, 4, 0], [0, 0, 1]]
    
    expected_cov = jnp.diag(jnp.array([1.0, 4.0, 1.0]))
    assert jnp.allclose(cov[0], expected_cov, atol=1e-5)
