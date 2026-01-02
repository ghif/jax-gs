import jax
import jax.numpy as jnp
import numpy as np
from jax_gs.core.gaussians import Gaussians, init_gaussians_from_pcd, get_covariance_3d
from jax_gs.core.camera import Camera

def test_gaussians_initialization():
    num_points = 100
    points = np.random.uniform(-1, 1, (num_points, 3))
    colors = np.random.uniform(0, 1, (num_points, 3))
    
    gaussians = init_gaussians_from_pcd(jnp.array(points), jnp.array(colors))
    
    assert isinstance(gaussians, Gaussians)
    assert gaussians.means.shape == (num_points, 3)
    assert gaussians.scales.shape == (num_points, 3)
    assert gaussians.quaternions.shape == (num_points, 4)
    assert gaussians.opacities.shape == (num_points, 1)
    assert gaussians.sh_coeffs.shape == (num_points, 16, 3)

def test_covariance_3d_properties():
    num_points = 10
    scales = jnp.array(np.random.uniform(-1, 1, (num_points, 3)))
    # Random quaternions (normalized)
    quats = jnp.array(np.random.uniform(-1, 1, (num_points, 4)))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    
    cov3D = get_covariance_3d(scales, quats)
    
    assert cov3D.shape == (num_points, 3, 3)
    
    # Check for symmetry
    for i in range(num_points):
        np.testing.assert_allclose(cov3D[i], cov3D[i].T, atol=1e-6)
        
        # Check for positive semi-definiteness (eigenvalues >= 0)
        eigenvalues = jnp.linalg.eigvalsh(cov3D[i])
        assert jnp.all(eigenvalues >= -1e-7)

def test_camera_entity():
    cam = Camera(
        W=640, H=480,
        fx=500.0, fy=500.0,
        cx=320.0, cy=240.0,
        W2C=jnp.eye(4),
        full_proj=jnp.eye(4)
    )
    assert cam.W == 640
    assert cam.H == 480
    assert cam.fx == 500.0
    assert cam.W2C.shape == (4, 4)
