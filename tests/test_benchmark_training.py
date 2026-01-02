import jax
import jax.numpy as jnp
import optax
import time
import os
import pytest
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.training.trainer import train_step

def test_benchmark_training_fern():
    """
    Benchmark a single training iteration using the Fern dataset.
    This includes rendering, loss calculation, and backpropagation.
    """
    path = "data/nerf_example_data/nerf_llff_data/fern"
    if not os.path.exists(path):
        pytest.skip("Fern dataset not found at expected path.")
        
    # 1. Load Data
    print(f"\nLoading Fern dataset for training benchmark from {path}...")
    xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")
    
    num_points = xyz.shape[0]
    camera = jax_cameras[0]
    target = jax_targets[0]
    W, H = camera.W, camera.H
    
    # 2. Initialize Gaussians and Optimizer
    gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    
    # Prepare static camera args
    camera_static = (camera.W, camera.H, camera.fx, camera.fy, camera.cx, camera.cy)
    
    print(f"Benchmarking training step with {num_points} Gaussians at {W}x{H} resolution...")
    
    # JIT Warm-up (This will be slow as it compiles the whole grad function)
    print("Warming up (JIT compilation of train_step)...")
    start_warm = time.perf_counter()
    state, loss = train_step(state, target, camera.W2C, camera_static, optimizer)
    jax.block_until_ready(loss)
    end_warm = time.perf_counter()
    print(f"Warm-up/Compilation took {end_warm - start_warm:.4f}s")
    
    # Benchmark runs
    num_runs = 10
    print(f"Running benchmark ({num_runs} training iterations)...")
    
    times = []
    for i in range(num_runs):
        # Rotate through cameras if we want more variety, but for speed check one is fine
        cam_idx = i % len(jax_cameras)
        curr_cam = jax_cameras[cam_idx]
        curr_target = jax_targets[cam_idx]
        
        start = time.perf_counter()
        state, loss = train_step(state, curr_target, curr_cam.W2C, camera_static, optimizer)
        jax.block_until_ready(loss)
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Iteration {i+1}: {end - start:.4f}s (Loss: {float(loss):.4f})")
    
    # Skip iteration 1 for warmup
    times = times[1:]
    num_runs -= 1
        
    avg_time = sum(times) / num_runs
    it_per_sec = 1.0 / avg_time
    print(f"\nTraining Benchmark Result (Fern):")
    print(f"  Average Iteration Time: {avg_time:.4f}s")
    print(f"  Average Speed:          {it_per_sec:.2f} it/s")
    
    assert avg_time < 10.0 # Sanity check

if __name__ == "__main__":
    test_benchmark_training_fern()
