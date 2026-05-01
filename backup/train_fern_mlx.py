import mlx.core as mx
import mlx.optimizers as opt
import numpy as np
import os
import datetime
import random
import argparse
from tqdm import tqdm
from PIL import Image

from mlx_gs.core.gaussians import init_gaussians_from_pcd
from mlx_gs.io.colmap import load_colmap_dataset
from mlx_gs.training.trainer import Trainer

def run_training(num_iterations: int = 10000):
    # 1. Load Data
    path = "data/nerf_example_data/nerf_llff_data/fern"
    xyz, rgb, mlx_cameras, mlx_targets = load_colmap_dataset(path, "images_8")
    
    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(mlx_cameras)} cameras for training")
    
    # 2. Initialize Gaussians
    params = init_gaussians_from_pcd(xyz, rgb)
    
    # 3. Setup Trainer
    trainer = Trainer(params, lr=1e-3)
    
    # 4. Run Training
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"fern_mlx_{timestamp}")
    
    trainer.train(
        cameras=mlx_cameras,
        targets=mlx_targets,
        num_iterations=num_iterations,
        output_dir=output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=1000)
    args = parser.parse_args()
    
    run_training(num_iterations=args.num_iterations)
