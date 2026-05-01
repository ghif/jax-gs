import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.camera import Camera
from jax_gs.renderer.projection import project_gaussians
from jax_gs.renderer.rasterizer import get_tile_interactions, render_tiles, TILE_SIZE

def test_projection():
    # Setup a single gaussian
    means3D = jnp.array([[0.0, 0.0, 5.0]]) # 5 units ahead on Z
    colors = jnp.array([[1.0, 1.0, 1.0]])
    gaussians = init_gaussians_from_pcd(means3D, colors)
    
    # Simple pinhole camera
    W, H = 100, 100
    cam = Camera(
        W=W, H=H,
        fx=50.0, fy=50.0,
        cx=50.0, cy=50.0,
        W2C=jnp.eye(4),
        full_proj=jnp.eye(4)
    )
    
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, cam)
    
    # Should be centered
    assert jnp.allclose(means2D[0], jnp.array([50.0, 50.0]))
    assert valid_mask[0] == True
    assert depths[0] == 5.0
    assert radii[0] > 0

def test_tile_interactions():
    W, H = 32, 32
    means2D = jnp.array([[16.0, 16.0], [0.0, 0.0]])
    radii = jnp.array([4, 2], dtype=jnp.int32)
    valid_mask = jnp.array([True, True])
    depths = jnp.array([5.0, 10.0])
    
    sorted_tile_ids, sorted_gaussian_ids, n_interactions = get_tile_interactions(
        means2D, radii, valid_mask, depths, H, W, TILE_SIZE
    )
    
    assert n_interactions > 0
    # Tile (1,1) should contain gaussian 0
    # Tiles are 16x16, so 32x32 has 2x2 tiles.
    # (16, 16) is at the corner of 4 tiles or center depending on indexing.
    # In our rasterizer, tile_id = (y // TILE_SIZE) * (W // TILE_SIZE) + (x // TILE_SIZE)
    assert len(sorted_tile_ids) > 0

def test_rasterization_op():
    # Setup mock data for rasterizer
    W, H = 32, 32
    means2D = jnp.array([[16.0, 16.0]])
    cov2D = jnp.array([[[10.0, 0.0], [0.0, 10.0]]])
    opacities = jnp.array([[10.0]]) # sigmoid(10) ~ 1
    colors = jnp.array([[1.0, 0.0, 0.0]]) # Red
    
    # Fake interactions (1 gaussian in 4 tiles)
    sorted_tile_ids = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    sorted_gaussian_ids = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    
    background = jnp.array([0.0, 0.0, 0.0])
    
    image = render_tiles(
        means2D, cov2D, opacities, colors,
        sorted_tile_ids, sorted_gaussian_ids,
        H, W, TILE_SIZE, background
    )
    
    assert image.shape == (H, W, 3)
    # Center should be red
    assert image[16, 16, 0] > 0.5
    assert image[16, 16, 1] < 0.1

def test_full_render():
    from jax_gs.renderer.renderer import render

    # 1. Setup a single centered Gaussian
    means3D = jnp.array([[0.0, 0.0, 5.0]]) # 5 units ahead
    colors = jnp.array([[1.0, 0.0, 0.0]]) # Red
    gaussians = init_gaussians_from_pcd(means3D, colors)
    
    # Set high opacity and small scale for a sharp point
    gaussians = gaussians.replace(
        opacities=jnp.array([[10.0]]), # Near 1.0 after sigmoid
        scales=jnp.array([[-1.0, -1.0, -1.0]]) # log(0.36) approx
    )
    
    # 2. Setup Camera
    W, H = 64, 64
    cam = Camera(
        W=W, H=H,
        fx=100.0, fy=100.0,
        cx=32.0, cy=32.0,
        W2C=jnp.eye(4),
        full_proj=jnp.eye(4)
    )
    
    # 3. Render
    image, extras = render(gaussians, cam)
    
    # NEW: Save image for visual inspection
    from PIL import Image
    import numpy as np
    img_np = (np.array(image) * 255).astype(np.uint8)
    assert img_np.shape == (H, W, 3)
    
    # Center should be red
    assert img_np[32, 32, 0] > 200
    assert img_np[32, 32, 1] < 50
