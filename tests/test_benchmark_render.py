import jax
import jax.numpy as jnp
import numpy as np
import time
import argparse
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.camera import Camera
from jax_gs.renderer.renderer import render

def test_benchmark_renderer(use_pallas=False, backend="gpu"):
    """
    Benchmark the JAX renderer performance.
    """
    # Check JAX version and device
    print(f"JAX Version: {jax.__version__}")
    print(f"JAX Devices: {jax.devices()}")
    print(f"Using Pallas: {use_pallas}")
    if use_pallas:
        print(f"Pallas Backend: {backend}")

    # 1. Setup Data (Scale of Fern Dataset @ 1/8 res)
    num_points = 10091
    W, H = 504, 378
    
    print(f"\nBenchmarking with {num_points} Gaussians at {W}x{H} resolution...")
    
    xyz = np.random.uniform(-1, 1, (num_points, 3))
    rgb = np.random.uniform(0, 1, (num_points, 3))
    gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
    
    cam = Camera(
        W=W, H=H,
        fx=400.0, fy=400.0,
        cx=W/2, cy=H/2,
        W2C=jnp.eye(4),
        full_proj=jnp.eye(4)
    )
    
    # JIT Compile the render function using a helper to handle static camera parameters
    print("Compiling render function (JIT)...")
    
    @jax.jit
    def jitted_render(gaussians, W2C, full_proj):
        # Create camera object with static W, H from closure
        # and dynamic W2C, full_proj from arguments
        curr_cam = Camera(
            W=W, H=H,
            fx=400.0, fy=400.0,
            cx=W/2, cy=H/2,
            W2C=W2C,
            full_proj=full_proj
        )
        return render(gaussians, curr_cam, fast_tpu_rasterizer=(backend == "tpu"))
    
    # JIT Warm-up
    print("Warming up (JIT)...")
    start_warm = time.perf_counter()
    image = jitted_render(gaussians, cam.W2C, cam.full_proj)
    jax.block_until_ready(image)
    end_warm = time.perf_counter()
    print(f"Warm-up took {end_warm - start_warm:.4f}s")
    
    # Benchmark runs
    num_runs = 20
    print(f"Running benchmark ({num_runs} iterations)...")
    
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        image = jitted_render(gaussians, cam.W2C, cam.full_proj)
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark JAX Gaussian Splatting renderer.")
    parser.add_argument("--pallas", type=str2bool, nargs='?', const=True, default=False, 
                        help="Use Pallas renderer (True/False or just --pallas for True).")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "tpu"],
                        help="Backend for Pallas (gpu or tpu).")
    args = parser.parse_args()

    # If running directly, execute the benchmark
    test_benchmark_renderer(use_pallas=args.pallas, backend=args.backend)
