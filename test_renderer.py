import jax
import jax.numpy as jnp
from renderer import Camera, render
from gaussians import Gaussians, init_gaussians_from_pcd

def test_renderer():
    W, H = 64, 64
    num_points = 10
    points = jnp.zeros((num_points, 3))
    colors = jnp.ones((num_points, 3))
    gaussians = init_gaussians_from_pcd(points, colors)
    
    cam = Camera(
        W=W, H=H,
        fx=50.0, fy=50.0,
        cx=W/2, cy=H/2,
        W2C=jnp.eye(4),
        full_proj=jnp.eye(4)
    )
    
    print("Rendering...")
    image = render(gaussians, cam)
    print("Shape:", image.shape)
    print("Values range:", jnp.min(image), jnp.max(image))
    assert image.shape == (H, W, 3)
    print("Test passed!")

if __name__ == "__main__":
    test_renderer()
