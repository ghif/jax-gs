import jax
import optax
import numpy as np
import os
import datetime
import random
import argparse
from tqdm import tqdm
from PIL import Image

from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.renderer.renderer import render
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.io.ply import save_ply
from jax_gs.training.trainer import train_step

def run_training(num_iterations: int = 10000):
    # 1. Load Data
    path = "data/nerf_example_data/nerf_llff_data/fern"
    xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")
    
    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(jax_cameras)} cameras for training")
    
    # 2. Initialize Gaussians
    gaussians = init_gaussians_from_pcd(np.array(xyz), np.array(rgb))
    
    # 3. Setup Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    
    # 4. Training Loop
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"fern_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    progress_dir = os.path.join(output_dir, "progress")
    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    pbar = tqdm(range(num_iterations))
    
    for i in pbar:
        # Pick random camera
        idx = random.randint(0, len(jax_cameras)-1)
        cam = jax_cameras[idx]
        target = jax_targets[idx]
        
        # Static args for JIT
        camera_static = (int(cam.W), int(cam.H), float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy))
        
        state, loss = train_step(state, target, cam.W2C, camera_static, optimizer)
        
        if i % 10 == 0:
            pbar.set_description(f"Loss: {loss:.4f}")
            
            if i % 100 == 0:
                # Render logic
                img = render(state[0], jax_cameras[0])
                img_np = np.array(img)
                Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(progress_dir, f"progress_{i:04d}.png")
                )
                save_ply(os.path.join(ply_dir, f"fern_splats_{i:04d}.ply"), state[0]) 

    # Final Save
    print("Training done. Saving final model...")
    save_ply(os.path.join(output_dir, "fern_final.ply"), state[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=10000)
    args = parser.parse_args()
    
    run_training(num_iterations=args.num_iterations)
