import jax
import jax.numpy as jnp
import numpy as np
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.camera import Camera
from jax_gs.renderer.projection import project_gaussians
from jax_gs.renderer.rasterizer import get_tile_interactions, TILE_SIZE

def test_project_gaussians():
    # Identity setup
    means3D = jnp.array([[0.0, 0.0, 5.0]]) # 5 units ahead
    scales = jnp.zeros((1, 3)) # log(1) = 0
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    opacities = jnp.zeros((1, 1))
    sh_coeffs = jnp.zeros((1, 16, 3))
    
    # Manually build Gaussians
    from jax_gs.core.gaussians import Gaussians
    gaussians = Gaussians(
        means=means3D,
        scales=scales,
        quaternions=quats,
        opacities=opacities,
        sh_coeffs=sh_coeffs
    )
    
    cam = Camera(
        W=100, H=100,
        fx=100.0, fy=100.0,
        cx=50.0, cy=50.0,
        W2C=jnp.eye(4),
        full_proj=jnp.eye(4)
    )
    
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, cam)
    
    # With identity W2C and point at (0,0,5), 2D mean should be at center (50, 50)
    # Projection: x_img = fx * 0/5 + cx = 50, y_img = fy * 0/5 + cy = 50
    np.testing.assert_allclose(means2D[0], jnp.array([50.0, 50.0]), atol=1e-5)
    assert valid_mask[0] == True
    assert depths[0] == 5.0

def test_project_gaussians_clipping():
    # Point behind camera
    means3D = jnp.array([[0.0, 0.0, -5.0]])
    gaussians = init_gaussians_from_pcd(means3D, jnp.zeros((1, 3)))
    
    cam = Camera(
        W=100, H=100, fx=100.0, fy=100.0, cx=50.0, cy=50.0,
        W2C=jnp.eye(4), full_proj=jnp.eye(4)
    )
    
    _, _, _, valid_mask, _ = project_gaussians(gaussians, cam)
    assert valid_mask[0] == False

def test_bitpacked_sort_robustness():
    # Test that get_tile_interactions correctly sorts by tile_id then depth
    # without running into int32 overflow or sentinel issues.
    
    W, H = 100, 100
    # Two points in same tile (say tile 0) at different depths
    means2D = jnp.array([[5.0, 5.0], [6.0, 6.0], [500.0, 500.0]]) # Point 3 is out of bounds
    radii = jnp.array([2.0, 2.0, 2.0])
    valid_mask = jnp.array([True, True, True])
    depths = jnp.array([10.0, 5.0, 1.0]) # Point 1 is deeper than point 2
    
    sorted_tiles, sorted_ids, count = get_tile_interactions(
        means2D, radii, valid_mask, depths, H, W, TILE_SIZE
    )
    
    # Total interactions should be 2 (point 3 is out of range)
    assert count == 2
    
    # First valid IDs in sorted list should be [1, 0] because depths[1]=5.0 < depths[0]=10.0
    # and they are in the same tile (tile 0).
    # Wait, the list is num_points * 64, so we look at the first two valid ones.
    
    # Tile IDs should be 0 for both
    assert sorted_tiles[0] == 0
    assert sorted_tiles[1] == 0
    
    # Gaussian IDs should be sorted by depth
    assert sorted_ids[0] == 1 # Depth 5.0
    assert sorted_ids[1] == 0 # Depth 10.0
    
    # Check that invalid ones (sentinels) are at the end
    # Num tiles = (100//16+1)^2 = 7*7 = 49
    # The sentinel we used is num_tiles_total
    num_tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
    sentinel = num_tiles_x * num_tiles_y
    
    assert sorted_tiles[-1] == sentinel
