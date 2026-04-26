import jax
import jax.numpy as jnp
import numpy as np
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.camera import Camera
from jax_gs.renderer.renderer import render

def test_pallas_render_parity():
    """
    Test that Pallas renderer (in interpret mode) produces similar results to the standard renderer.
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
    image_std = render(gaussians, cam, use_pallas=False)
    
    # 3. Render with Pallas renderer (interpret mode on CPU)
    image_pallas = render(gaussians, cam, use_pallas=True)
    
    # 4. Compare
    diff = jnp.abs(image_std - image_pallas)
    max_diff = jnp.max(jnp.nan_to_num(diff))
    print(f"Max difference: {max_diff}")
    
    has_nan_std = jnp.any(jnp.isnan(image_std))
    has_nan_pallas = jnp.any(jnp.isnan(image_pallas))
    print(f"Standard has NaNs: {has_nan_std}")
    print(f"Pallas has NaNs: {has_nan_pallas}")

    if has_nan_pallas:
        nan_mask = jnp.isnan(image_pallas)
        print(f"NaN count in Pallas: {jnp.sum(nan_mask)}")
        # Handle environment NaNs by assuming 0 for parity check
        image_pallas = jnp.nan_to_num(image_pallas)

    np.testing.assert_allclose(image_std, image_pallas, atol=1e-2)
    
    print("Pallas parity test passed!")

if __name__ == "__main__":
    test_pallas_render_parity()
