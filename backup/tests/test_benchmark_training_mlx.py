import mlx.core as mx
import mlx.optimizers as opt
import numpy as np
import time
import os
import pytest
import random
from mlx_gs.core.gaussians import init_gaussians_from_pcd
from mlx_gs.io.colmap import load_colmap_dataset
from mlx_gs.training.trainer import Trainer

def test_benchmark_training_mlx_fern():
    """
    Benchmark a single training iteration using the Fern dataset (MLX mode).
    This includes rendering, loss calculation, and backpropagation.
    """
    path = "data/nerf_example_data/nerf_llff_data/fern"
    if not os.path.exists(path):
        pytest.skip("Fern dataset not found at expected path.")
        
    # 1. Load Data
    print(f"\nLoading Fern dataset for MLX training benchmark from {path}...")
    xyz, rgb, mlx_cameras, mlx_targets = load_colmap_dataset(path, "images_8")
    
    num_points = xyz.shape[0]
    W, H = mlx_cameras[0]["W"], mlx_cameras[0]["H"]
    
    # 2. Initialize Gaussians and Trainer
    params = init_gaussians_from_pcd(xyz, rgb)
    trainer = Trainer(params, lr=1e-3)
    
    print(f"Benchmarking MLX training step with {num_points} Gaussians at {W}x{H} resolution...")
    
    # Warm-up (Force computation with eval)
    print("Warming up (initial iteration and compilation if any)...")
    start_warm = time.perf_counter()
    _ = trainer.train_step(mlx_cameras[0], mlx_targets[0])
    end_warm = time.perf_counter()
    print(f"Warm-up took {end_warm - start_warm:.4f}s")
    
    # Benchmark runs
    num_runs = 10
    print(f"Running benchmark ({num_runs} training iterations)...")
    
    times = []
    for i in range(num_runs):
        # Rotate through cameras
        cam_idx = i % len(mlx_cameras)
        curr_cam = mlx_cameras[cam_idx]
        curr_target = mlx_targets[cam_idx]
        
        start = time.perf_counter()
        loss = trainer.train_step(curr_cam, curr_target)
        end = time.perf_counter()
        
        times.append(end - start)
        print(f"  Iteration {i+1}: {end - start:.4f}s (Loss: {loss:.4f})")
    
    # Skip iteration 1 for more consistent average (lazy compilation/init in MLX)
    times = times[1:]
    num_runs -= 1
        
    avg_time = sum(times) / num_runs
    it_per_sec = 1.0 / avg_time
    print(f"\nTraining Benchmark Result (Fern - MLX):")
    print(f"  Average Iteration Time: {avg_time:.4f}s")
    print(f"  Average Speed:          {it_per_sec:.2f} it/s")
    
    assert avg_time < 10.0 # Sanity check

if __name__ == "__main__":
    test_benchmark_training_mlx_fern()
