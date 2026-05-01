import jax
# Force JAX to CPU early
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import numpy as np
import time
import sys
import os

from jax_gs.core.gaussians import Gaussians
from jax_gs.core.camera import Camera
from jax_gs.renderer.renderer import render

def benchmark():
    print(f"Python Executable: {sys.executable}")
    # print(f"Python Path: {sys.path}")
    
    # Setup dummy data
    num_gaussians = 100000
    H, W = 1080, 1920
    
    means = np.random.normal(0, 1, (num_gaussians, 3)).astype(np.float32)
    sh_coeffs = np.random.uniform(0, 1, (num_gaussians, 1, 3)).astype(np.float32)
    opacities = np.random.uniform(0.1, 0.9, (num_gaussians, 1)).astype(np.float32)
    scales = np.random.uniform(-3, -1, (num_gaussians, 3)).astype(np.float32)
    quats = np.random.normal(0, 1, (num_gaussians, 4)).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    
    cpu_device = jax.devices("cpu")[0]
    
    gaussians = Gaussians(
        means=jax.device_put(jnp.array(means), cpu_device),
        sh_coeffs=jax.device_put(jnp.array(sh_coeffs), cpu_device),
        opacities=jax.device_put(jnp.array(opacities), cpu_device),
        scales=jax.device_put(jnp.array(scales), cpu_device),
        quaternions=jax.device_put(jnp.array(quats), cpu_device)
    )
    
    camera = Camera(
        H=H, W=W, fx=1000.0, fy=1000.0, cx=float(W/2), cy=float(H/2),
        W2C=jax.device_put(jnp.eye(4), cpu_device),
        full_proj=jax.device_put(jnp.eye(4), cpu_device)
    )
    
    print(f"Benchmarking with {num_gaussians} Gaussians at {W}x{H}")
    print(f"JAX Device: {gaussians.means.device}")

    # 1. Warm-up JAX
    print("Warming up JAX...")
    _ = render(gaussians, camera, use_mlx=False)
    
    # 2. Benchmark JAX
    print("Benchmarking JAX...")
    start = time.time()
    for _ in range(10):
        img_jax = render(gaussians, camera, use_mlx=False)
        jax.block_until_ready(img_jax)
    end = time.time()
    print(f"JAX Average Time: {(end - start) / 10:.4f}s")
    
    # 3. Benchmark MLX (if available)
    try:
        import mlx.core as mx
        print("Warming up MLX...")
        _ = render(gaussians, camera, use_mlx=True)
        
        print("Benchmarking MLX...")
        start = time.time()
        for _ in range(10):
            img_mlx = render(gaussians, camera, use_mlx=True)
            # MLX is lazy, so we need to eval
            mx.eval(img_mlx)
        end = time.time()
        print(f"MLX Average Time: {(end - start) / 10:.4f}s")
        
        # 4. Correctness Check (sample)
        img_jax_np = np.array(img_jax)
        img_mlx_np = np.array(img_mlx)
        diff = np.abs(img_jax_np - img_mlx_np).mean()
        print(f"Mean Pixel Difference: {diff:.6f}")
        
    except ImportError:
        print("MLX not found. Please install it to benchmark.")

    # 4. Benchmark PyTorch (if available)
    try:
        import torch
        print("Warming up PyTorch (MPS)...")
        _ = render(gaussians, camera, use_torch=True)
        
        print("Benchmarking PyTorch (MPS)...")
        start = time.time()
        for _ in range(10):
            img_torch = render(gaussians, camera, use_torch=True)
            # Torch MPS is asynchronous, but calling .cpu() or similar forces sync
            _ = img_torch.sum()
        end = time.time()
        print(f"PyTorch (MPS) Average Time: {(end - start) / 10:.4f}s")
        
        # Correctness Check
        img_jax_np = np.array(img_jax)
        img_torch_np = np.array(img_torch)
        diff = np.abs(img_jax_np - img_torch_np).mean()
        print(f"PyTorch Mean Pixel Difference: {diff:.6f}")
        
    except ImportError:
        print("PyTorch not found. Please install it to benchmark.")

if __name__ == "__main__":
    benchmark()
