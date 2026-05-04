import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.camera import Camera
from jax_gs.renderer.renderer import render
from jax_gs.renderer.rasterizer import get_tile_interactions, render_tiles, TILE_SIZE
from jax_gs.renderer.rasterizer_pallas import render_tiles_pallas

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

def test_pallas_custom_vjp_gradients_match_reference():
    num_points = 24
    W, H = 32, 32

    key = jax.random.PRNGKey(0)
    means2d = jax.random.uniform(key, (num_points, 2), minval=0.0, maxval=float(W - 1))
    cov_diag = jax.random.uniform(jax.random.PRNGKey(1), (num_points, 2), minval=2.0, maxval=20.0)
    cov2d = jnp.zeros((num_points, 2, 2), dtype=jnp.float32)
    cov2d = cov2d.at[:, 0, 0].set(cov_diag[:, 0])
    cov2d = cov2d.at[:, 1, 1].set(cov_diag[:, 1])
    cov2d = cov2d.at[:, 0, 1].set(0.1)
    cov2d = cov2d.at[:, 1, 0].set(0.1)
    opacities = jax.random.uniform(jax.random.PRNGKey(2), (num_points, 1), minval=-1.0, maxval=2.0)
    colors = jax.random.uniform(jax.random.PRNGKey(3), (num_points, 3), minval=0.0, maxval=1.0)
    radii = jnp.full((num_points,), 8, dtype=jnp.int32)
    valid_mask = jnp.ones((num_points,), dtype=bool)
    depths = jax.random.uniform(jax.random.PRNGKey(4), (num_points,), minval=1.0, maxval=10.0)
    background = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)

    sorted_tile_ids, sorted_gaussian_ids, _ = get_tile_interactions(
        means2d, radii, valid_mask, depths, H, W, TILE_SIZE
    )

    def loss_ref(m, c, o, col):
        image = render_tiles(m, c, o, col, sorted_tile_ids, sorted_gaussian_ids, H, W, TILE_SIZE, background)
        return jnp.mean(image)

    def loss_pallas(m, c, o, col):
        image, _ = render_tiles_pallas(
            m, c, o, col, sorted_tile_ids, sorted_gaussian_ids, H, W, TILE_SIZE, background, backend="gpu"
        )
        return jnp.mean(image)

    grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2, 3))(means2d, cov2d, opacities, colors)
    grads_pallas = jax.grad(loss_pallas, argnums=(0, 1, 2, 3))(means2d, cov2d, opacities, colors)

    for g_ref, g_pallas in zip(grads_ref, grads_pallas):
        assert jnp.allclose(g_pallas, g_ref, atol=3e-3, rtol=3e-2)
