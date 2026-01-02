
import jax
import optax
import numpy as np
import os
import datetime
import random
import argparse
import glob
from tqdm import tqdm
from functools import partial
from PIL import Image

from jax_gs.core.gaussians import Gaussians
from jax_gs.renderer.renderer import render
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.io.ply import save_ply, load_ply
from jax_gs.training.trainer import train_step

def find_latest_ply():
    ply_files = glob.glob("results/fern_*/ply/*.ply")
    if not ply_files:
        return None
    # Sort by modification time
    ply_files.sort(key=os.path.getmtime)
    return ply_files[-1]

def run_resume_training(num_iterations: int = 10000):
    # 1. Load Data
    path = "data/nerf_example_data/nerf_llff_data/fern"
    _, _, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")
    
    # 2. Find and Load Latest PLY
    latest_ply = find_latest_ply()
    if not latest_ply:
        print("No existing PLY found to resume.")
        return
    
    print(f"Resuming from: {latest_ply}")
    gaussians = load_ply(latest_ply)
    
    # 3. Setup Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    
    # 4. Training Loop
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"fern_resume_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    progress_dir = os.path.join(output_dir, "progress")
    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    # Infer start iteration
    try:
        start_iter = int(os.path.basename(latest_ply).split('_')[-1].split('.')[0])
    except:
        start_iter = 0
    
    print(f"Starting from iteration {start_iter}")
    pbar = tqdm(range(num_iterations))
    
    for i in pbar:
        curr_iter = start_iter + i
        idx = random.randint(0, len(jax_cameras)-1)
        cam = jax_cameras[idx]
        target = jax_targets[idx]
        
        camera_static = (int(cam.W), int(cam.H), float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy))
        state, loss = train_step(state, target, cam.W2C, camera_static, optimizer)
        
        if i % 10 == 0:
            pbar.set_description(f"Iter {curr_iter} | Loss: {loss:.4f}")
            
            if i % 100 == 0:
                img = render(state[0], jax_cameras[0])
                img_np = np.array(img)
                Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(progress_dir, f"progress_{curr_iter:04d}.png")
                )
                save_ply(os.path.join(ply_dir, f"fern_splats_{curr_iter:04d}.ply"), state[0]) 

    print("Training done. Saving final model...")
    save_ply(os.path.join(output_dir, "fern_final_resume.ply"), state[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=10000)
    args = parser.parse_args()
    
    run_resume_training(num_iterations=args.num_iterations)
