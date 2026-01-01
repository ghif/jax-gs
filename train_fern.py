
import jax
# Configure JAX to use CPU or GPU as available (defaulting to CPU for safety unless user has GPU)
# The user env is simple, lets rely on JAX default or force CPU if needed.
# jax.config.update('jax_platform_name', 'cpu') # Uncomment if CPU only is desired

import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
import os
import time
from PIL import Image
import datetime
import random

# from renderer import Camera, render
from renderer_v2 import Camera, render_v2 as render
from gaussians import Gaussians, init_gaussians_from_pcd
from utils import load_colmap_data
from save_utils import save_ply

def l1_loss(pred, target):
    """
    Args:
        pred: (H, W, 3) array of predicted colors
        target: (H, W, 3) array of target colors
    Returns:
        loss: float, mean absolute error between predicted and target colors
    """
    return jnp.mean(jnp.abs(pred - target))

def ssim_loss(pred, target):
    """
    Args:
        pred: (H, W, 3) array of predicted colors
        target: (H, W, 3) array of target colors
    Returns:
        loss: float, mean absolute error between predicted and target colors
    """
    # Simplified SSIM or just use L1 for now?
    # Using simple L1 + D-SSIM is standard for GS.
    # For this simplified implementation, we stick to L1 to avoid complexity unless requested.
    return l1_loss(pred, target)

def create_camera_objects(train_cameras):
    cameras = []
    for cam in train_cameras:
        # Construct W2C matrix
        w2c = np.eye(4)
        w2c[:3, :3] = cam.R
        w2c[:3, 3] = cam.T
        
        # In renderer.py, W2C is expected to be consistent with how points are transformed.
        # means_cam = (means_homo @ W2C.T)
        # So W2C should be the matrix that transforms World to Camera.
        # cam.R and cam.T are typically World-to-Camera in COLMAP.
        # So this should be correct.
        
        c = Camera(
            W=cam.width,
            H=cam.height,
            fx=float(cam.FovX), # Wait, FovX in utils is FOV in radians. fx in renderer is focal length.
            fy=float(cam.FovY), # Wait, logic in utils calculated fov from f.
            cx=cam.width/2.0,   # Approx if not stored? 
            # In utils.py I stored (fx, fy, cx, cy) in params! I should use that because FovX/Y were derived.
            cy=cam.height/2.0,
            W2C=jnp.array(w2c),
            full_proj=jnp.eye(4) # Unused
        )
        
        # Correctly extracting fx, fy, cx, cy from cam.params if I exposed it.
        # CameraInfo uses FovX/FovY but seemingly discarded fx/fy in my utils definition?
        # Let's check Utils.CameraInfo definition.
        
        cameras.append(c)
    return cameras

# I need to fix Utils or use what I have.
# In utils.py: 
# train_cameras.append(CameraInfo(..., FovY=fovy, FovX=fovx, ...))
# I calculated fovx/fovy from params. I did NOT store params in CameraInfo!!
# I should probably just reverse it or better yet, store proper intrinsics in CameraInfo.
# But `renderer.Camera` expects `fx`, `fy`.
# `fx = W / (2 * tan(FovX / 2))`
# So I can reconstruct it.

@jax.jit(static_argnums=(3, 4))
def train_step(state, target_image, w2c, optimizer, camera_static):
    params, opt_state = state
    W, H, fx, fy, cx, cy = camera_static
    
    # Reconstruct Camera object with static intrinsics
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    def loss_fn(p):
        image = render(p, camera)
        loss = l1_loss(image, target_image)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    
    return (next_params, next_opt_state), loss

def run_training():
    path = "data/nerf_example_data/nerf_llff_data/fern"
    print(f"Loading data from {path}")
    xyz, rgb, train_cam_infos = load_colmap_data(path, "images_8")
    
    # Filter bounds?
    print(f"Loaded {len(xyz)} points")
    
    # Create JAX Camera objects
    # Recompute intrinsics from FOV
    jax_cameras = []
    jax_targets = []
    
    for info in train_cam_infos:
        fx = info.width / (2 * jnp.tan(info.FovX / 2))
        fy = info.height / (2 * jnp.tan(info.FovY / 2))
        cx = info.width / 2.0
        cy = info.height / 2.0
        
        # W2C
        w2c = np.eye(4)
        w2c[:3, :3] = info.R
        w2c[:3, 3] = info.T
        
        c = Camera(
            W=info.width,
            H=info.height,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            W2C=jnp.array(w2c),
            full_proj=jnp.eye(4)
        )
        jax_cameras.append(c)
        jax_targets.append(jnp.array(info.image))
        
    print(f"Prepared {len(jax_cameras)} cameras for training")
    
    # Initialize Gaussians
    gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
    
    # Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    
    # Loop
    num_iterations = 10000 # Short run for verification
    pbar = tqdm(range(num_iterations))
    
    
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"fern_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    progress_dir = os.path.join(output_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)

    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)
    
    for i in pbar:
        # Pick random camera
        idx = random.randint(0, len(jax_cameras)-1)
        cam = jax_cameras[idx]
        target = jax_targets[idx]
        
        # Deconstruct for JIT static args
        # Ensure values are simple python types (int, float) for static arguments
        camera_static = (int(cam.W), int(cam.H), float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy))
        w2c = cam.W2C
        
        state, loss = train_step(state, target, w2c, optimizer, camera_static)
        
        if i % 10 == 0:
            pbar.set_description(f"Loss: {loss:.4f}")
            
            print(f"\nEvaluating at step {i}...")
            img = render(state[0], jax_cameras[0])
            img_np = np.array(img)
            print(f"Image stats - Min: {img_np.min():.4f}, Max: {img_np.max():.4f}, Mean: {img_np.mean():.4f}")
            Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(os.path.join(progress_dir, f"progress_{i:04d}.png"))

            # Save PLY for Viser
            save_ply(os.path.join(ply_dir, f"fern_splats_{i:04d}.ply"), state[0]) 
    # end for

    # Save verification content
    print("Training done. Rendering view 0...")
    final_params, _ = state
    img = render(final_params, jax_cameras[0])
    img_np = np.array(img)
    Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(os.path.join(output_dir, "train_render_0.png"))
    
    # Save target for comparison
    tgt_np = np.array(jax_targets[0]) * 255.0
    Image.fromarray(tgt_np.astype(np.uint8)).save(os.path.join(output_dir, "target_0.png"))

    # Save PLY for Viser
    save_ply(os.path.join(ply_dir, "fern_final_splats.ply"), final_params)
    print("Saved fern_final_splats.ply")
    

if __name__ == "__main__":
    run_training()
