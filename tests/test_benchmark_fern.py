import jax
import jax.numpy as jnp
import time
import os
import pytest
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.renderer.renderer import render

def test_benchmark_renderer_fern():
    """
    Benchmark the JAX renderer performance using the Fern dataset.
    """
    path = "data/nerf_example_data/nerf_llff_data/fern"
    if not os.path.exists(path):
        pytest.skip("Fern dataset not found at expected path.")
        
    # 1. Load Data
    print(f"\nLoading Fern dataset from {path}...")
    xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")
    
    num_points = xyz.shape[0]
    camera = jax_cameras[0]
    W, H = camera.W, camera.H
    
    print(f"Benchmarking with {num_points} Gaussians at {W}x{H} resolution...")
    
    # 2. Initialize Gaussians
    gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
    
    # JIT Warm-up
    print("Warming up (JIT)...")
    start_warm = time.perf_counter()
    image = render(gaussians, camera)
    jax.block_until_ready(image)
    end_warm = time.perf_counter()
    print(f"Warm-up took {end_warm - start_warm:.4f}s")
    
    # Benchmark runs
    num_runs = 5
    print(f"Running benchmark ({num_runs} iterations)...")
    
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        image = render(gaussians, camera)
        jax.block_until_ready(image)
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Run {i+1}: {end - start:.4f}s")
    
    avg_time = sum(times) / num_runs
    fps = 1.0 / avg_time
    print(f"\nBenchmark Result (Fern):")
    print(f"  Average Time: {avg_time:.4f}s")
    print(f"  Average FPS:  {fps:.2f}")
    
    assert avg_time < 5.0

if __name__ == "__main__":
    test_benchmark_renderer_fern()
