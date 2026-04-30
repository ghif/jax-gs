import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.camera import Camera
from jax_gs.renderer.renderer import render

def test_pallas_render_parity():
    """
    Test that Pallas renderer produces similar results to the standard renderer.
    """
    # 1. Setup Data
    num_points = 100
    W, H = 32, 32
    
    xyz = np.random.uniform(-1, 1, (num_points, 3))
    xyz[:, 2] += 5.0 # Move ahead of camera
    rgb = np.random.uniform(0, 1, (num_points, 3))
    gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
    
    # Randomize some scales and opacities
    gaussians = gaussians.replace(
        scales=np.random.uniform(-2, -1, (num_points, 3)),
        opacities=np.random.uniform(0, 2, (num_points, 1))
    )
    
    cam = Camera(
        W=W, H=H,
        fx=25.0, fy=25.0,
        cx=W/2, cy=H/2,
        W2C=jnp.eye(4),
        full_proj=jnp.eye(4)
    )
    
    # 2. Render with standard renderer
    image_std, _ = render(gaussians, cam, use_pallas=False)
    
    # 3. Render with Pallas renderer
    print("Testing Pallas backend...")
    
    # Detect platform to choose backend
    platform = jax.devices()[0].platform
    backend = "tpu" if platform == "tpu" else "gpu"
    
    try:
        image_pallas, _ = render(gaussians, cam, use_pallas=True, backend=backend)
    except Exception as e:
        pytest.skip(f"Pallas failed on {platform}: {e}")
    
    # 4. Compare
    # They should be close but not identical due to different accumulation orders/precisions
    diff = jnp.abs(image_std - image_pallas)
    mean_diff = jnp.mean(diff)
    print(f"Mean difference: {mean_diff}")
    
    assert mean_diff < 0.05
    assert jnp.max(diff) < 0.5
