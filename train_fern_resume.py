
import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
import os
import datetime
import random
import math
import glob
from functools import partial
from PIL import Image
import argparse

# Use the standard renderer
from renderer_v2 import Camera, render_v2 as render
from gaussians import Gaussians
from utils import load_colmap_data
from ply_utils import save_ply, load_ply

def l1_loss(pred, target):
    return jnp.mean(jnp.abs(pred - target))

@partial(jax.jit, static_argnums=(3, 4))
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

def find_latest_ply():
    ply_files = glob.glob("results/fern_*/ply/*.ply")
    if not ply_files:
        return None
    # Sort by modification time
    ply_files.sort(key=os.path.getmtime)
    return ply_files[-1]

def run_training(num_iterations: int = 10000):
    # 1. Load data
    path = "data/nerf_example_data/nerf_llff_data/fern"
    print(f"Loading data from {path}")
    _, _, train_cam_infos = load_colmap_data(path, "images_8")
    
    jax_cameras = []
    jax_targets = []
    
    for info in train_cam_infos:
        fx = info.width / (2 * math.tan(info.FovX / 2))
        fy = info.height / (2 * math.tan(info.FovY / 2))
        cx = info.width / 2.0
        cy = info.height / 2.0
        
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
    
    # 2. Find and load latest PLY
    latest_ply = find_latest_ply()
    if not latest_ply:
        print("No existing PLY found. Cannot resume.")
        return
    
    print(f"Resuming from: {latest_ply}")
    gaussians = load_ply(latest_ply)
    
    # 3. Setup training
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    
    # Loop
    pbar = tqdm(range(num_iterations))
    
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"fern_resume_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    progress_dir = os.path.join(output_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)

    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)
    
    # Extract starting iteration from filename if possible (e.g., fern_splats_3900.ply)
    try:
        start_iter = int(os.path.basename(latest_ply).split('_')[-1].split('.')[0])
    except:
        start_iter = 0
    
    print(f"Starting from iteration {start_iter}")

    for i in pbar:
        iter_count = start_iter + i
        
        # Pick random camera
        idx = random.randint(0, len(jax_cameras)-1)
        cam = jax_cameras[idx]
        target = jax_targets[idx]
        
        camera_static = (int(cam.W), int(cam.H), float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy))
        w2c = cam.W2C
        
        state, loss = train_step(state, target, w2c, optimizer, camera_static)
        
        if i % 10 == 0:
            pbar.set_description(f"Iter {iter_count} | Loss: {loss:.4f}")
            
            if i % 100 == 0:
                img = render(state[0], jax_cameras[0])
                img_np = np.array(img)
                Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(progress_dir, f"progress_{iter_count:04d}.png")
                )
    
                save_ply(os.path.join(ply_dir, f"fern_splats_{iter_count:04d}.ply"), state[0]) 

    print("Training done. Rendering view 0...")
    final_params, _ = state
    img = render(final_params, jax_cameras[0])
    img_np = np.array(img)
    Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(os.path.join(output_dir, "train_render_0.png"))
    
    tgt_np = np.array(jax_targets[0]) * 255.0
    Image.fromarray(tgt_np.astype(np.uint8)).save(os.path.join(output_dir, "target_0.png"))

    save_ply(os.path.join(ply_dir, "fern_final_splats.ply"), final_params)
    print("Saved fern_final_splats.ply")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of training iterations to add")
    args = parser.parse_args()
    
    run_training(num_iterations=args.num_iterations)
