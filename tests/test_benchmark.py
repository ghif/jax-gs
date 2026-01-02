import jax
import jax.numpy as jnp
import numpy as np
import time
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.camera import Camera
from jax_gs.renderer.renderer import render

def test_benchmark_renderer():
    """
    Benchmark the JAX renderer performance.
    """
    # 1. Setup Data (Realistic Scale)
    num_points = 50_000
    W, H = 512, 512
    
    print(f"\nBenchmarking with {num_points} Gaussians at {W}x{H} resolution...")
    
    xyz = np.random.uniform(-1, 1, (num_points, 3))
    rgb = np.random.uniform(0, 1, (num_points, 3))
    gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
    
    cam = Camera(
        W=W, H=H,
        fx=500.0, fy=500.0,
        cx=W/2, cy=H/2,
        W2C=jnp.eye(4),
        full_proj=jnp.eye(4)
    )
    
    # JIT Warm-up
    print("Warming up (JIT)...")
    start_warm = time.perf_counter()
    image = render(gaussians, cam)
    jax.block_until_ready(image)
    end_warm = time.perf_counter()
    print(f"Warm-up took {end_warm - start_warm:.4f}s")
    
    # Benchmark runs
    num_runs = 5
    print(f"Running benchmark ({num_runs} iterations)...")
    
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        image = render(gaussians, cam)
        jax.block_until_ready(image)
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Run {i+1}: {end - start:.4f}s")
    
    avg_time = sum(times) / num_runs
    fps = 1.0 / avg_time
    print(f"\nBenchmark Result:")
    print(f"  Average Time: {avg_time:.4f}s")
    print(f"  Average FPS:  {fps:.2f}")
    
    assert avg_time < 5.0 # Sanity check for CPU, should be much faster on GPU

if __name__ == "__main__":
    # If running directly, execute the benchmark
    test_benchmark_renderer()
