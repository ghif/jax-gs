import jax
import jax.numpy as jnp
import pytest
from jax_gs.core.gaussians_2d import Gaussians2D, get_tangent_vectors, init_gaussians_2d_from_pcd
from jax_gs.renderer.projection_2d import project_gaussians_2d
from jax_gs.core.camera import Camera

def test_gaussians_2d_init():
    points = jnp.zeros((10, 3))
    colors = jnp.ones((10, 3))
    gaussians = init_gaussians_2d_from_pcd(points, colors)
    
    assert gaussians.means.shape == (10, 3)
    assert gaussians.scales.shape == (10, 2)
    assert gaussians.quaternions.shape == (10, 4)
    assert gaussians.opacities.shape == (10, 1)

def test_tangent_vectors():
    # Identity quaternion [1, 0, 0, 0] should result in 
    # u = [1, 0, 0], v = [0, 1, 0], n = [0, 0, 1]
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    u, v, n = get_tangent_vectors(quats)
    
    assert jnp.allclose(u, jnp.array([[1.0, 0.0, 0.0]]), atol=1e-5)
    assert jnp.allclose(v, jnp.array([[0.0, 1.0, 0.0]]), atol=1e-5)
    assert jnp.allclose(n, jnp.array([[0.0, 0.0, 1.0]]), atol=1e-5)
    
    # 90 degree rotation around X: [cos(45), sin(45), 0, 0] = [0.707, 0.707, 0, 0]
    # New Y should be Z, new Z should be -Y
    angle = jnp.pi / 2
    q_x = jnp.array([[jnp.cos(angle/2), jnp.sin(angle/2), 0.0, 0.0]])
    u, v, n = get_tangent_vectors(q_x)
    
    # u stays X: [1, 0, 0]
    # v was Y [0, 1, 0] -> becomes Z [0, 0, 1]
    # n was Z [0, 0, 1] -> becomes -Y [0, -1, 0]
    assert jnp.allclose(u, jnp.array([[1.0, 0.0, 0.0]]), atol=1e-5)
    assert jnp.allclose(v, jnp.array([[0.0, 0.0, 1.0]]), atol=1e-5)
    assert jnp.allclose(n, jnp.array([[0.0, -1.0, 0.0]]), atol=1e-5)

def test_projection_2d():
    gaussians = init_gaussians_2d_from_pcd(jnp.array([[0.0, 0.0, 5.0]]), jnp.ones((1, 3)))
    camera = Camera(
        W=100, H=100, fx=50, fy=50, cx=50, cy=50,
        W2C=jnp.eye(4), full_proj=jnp.eye(4)
    )
    
    means2D, cov2D, radii, valid_mask, z, normals = project_gaussians_2d(gaussians, camera)
    
    # Center is at [0, 0, 5]. Projected to image center [50, 50]
    assert jnp.allclose(means2D, jnp.array([[50.0, 50.0]]), atol=1e-5)
    assert z[0] == 5.0
    assert valid_mask[0] == True
    
    # Normal is [0, 0, 1] in world, [0, 0, 1] in camera (facing away from camera)
    assert jnp.allclose(normals, jnp.array([[0.0, 0.0, 1.0]]), atol=1e-5)
    
    # Covariance check (scale -3 in log space is exp(-3) = 0.049)
    # 2D Cov in image space should be around (fx/z * s)^2 + 0.3
    # (50/5 * 0.049)^2 + 0.3 = (0.49)^2 + 0.3 = 0.24 + 0.3 = 0.54
    assert jnp.allclose(cov2D[0, 0, 0], 0.54, atol=0.01)
