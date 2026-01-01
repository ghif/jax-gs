
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import time

import renderer_v2
from renderer_v2 import Camera
from gaussians import init_gaussians_from_pcd
from utils import load_colmap_data

def test_render():
    path = "data/nerf_example_data/nerf_llff_data/fern"
    print(f"Loading data from {path}")
    xyz, rgb, train_cam_infos = load_colmap_data(path, "images_8")
    
    # Take first camera
    info = train_cam_infos[0]
    
    fx = info.width / (2 * jnp.tan(info.FovX / 2))
    fy = info.height / (2 * jnp.tan(info.FovY / 2))
    cx = info.width / 2.0
    cy = info.height / 2.0
    
    w2c = np.eye(4)
    w2c[:3, :3] = info.R
    w2c[:3, 3] = info.T
    
    camera = Camera(
        W=int(info.width),
        H=int(info.height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        W2C=jnp.array(w2c),
        full_proj=jnp.eye(4)
    )
    
    print("Initializing Gaussians...")
    gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
    
    print("Compiling render_v2...")
    
    # We use render_camera_v2 and mark W (arg 6) and H (arg 7) as static
    # Args: gaussians, W2C, fx, fy, cx, cy, W, H, background
    render_jit = jax.jit(renderer_v2.render_camera_v2, static_argnums=(6, 7))
    
    start = time.time()
    img = render_jit(
        gaussians, 
        camera.W2C, 
        camera.fx, camera.fy, camera.cx, camera.cy, 
        camera.W, camera.H
    )
    img.block_until_ready()
    print(f"Compilation + First Run time: {time.time() - start:.4f}s")
    
    print("Second Run (Timing)...")
    start = time.time()
    img = render_jit(
        gaussians, 
        camera.W2C, 
        camera.fx, camera.fy, camera.cx, camera.cy, 
        camera.W, camera.H
    )
    img.block_until_ready()
    print(f"Second Run time: {time.time() - start:.4f}s")
    
    img_np = np.array(img)
    print(f"Output Stats: Min={img_np.min()}, Max={img_np.max()}, Mean={img_np.mean()}")
    
    Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save("test_render_v2.png")
    print("Saved test_render_v2.png")

if __name__ == "__main__":
    test_render()
