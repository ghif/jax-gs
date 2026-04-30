import jax
import jax.numpy as jnp
import optax
import time
import os
import pytest
import numpy as np
from PIL import Image
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.gaussians_2d import init_gaussians_2d_from_pcd
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.renderer.renderer import render
from jax_gs.training.trainer import train_step

def load_fern():
    import fsspec
    path = "gs://dataset-nerf/nerf_llff_data/fern"
    fs, _ = fsspec.core.url_to_fs(path)
    if not fs.exists(path):
        pytest.skip("Fern dataset not found at expected path.")
    return load_colmap_dataset(path, "images_8")

def benchmark_pallas(mode: str, xyz, rgb, jax_cameras, jax_targets):
    camera = jax_cameras[0]
    target = jax_targets[0]
    
    if mode == "2dgs":
        gaussians = init_gaussians_2d_from_pcd(jnp.array(xyz), jnp.array(rgb))
    else:
        gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
    
    # Detect platform
    platform = jax.devices()[0].platform
    backend = "tpu" if platform == "tpu" else "gpu"
    
    # --- Correctness Check ---
    print(f"Checking {mode} Pallas correctness vs JAX...")
    image_jax, extras_jax = render(gaussians, camera, mode=mode, use_pallas=False)
    image_pallas, extras_pallas = render(gaussians, camera, mode=mode, use_pallas=True, backend=backend)
    
    diff = jnp.abs(image_jax - image_pallas)
    mean_diff = float(jnp.mean(diff))
    max_diff = float(jnp.max(diff))
    print(f"  Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")
    
    if mean_diff > 0.05:
        print(f"  WARNING: {mode} Pallas has high difference vs JAX!")

    # --- Renderer Benchmark ---
    print(f"Warmup {mode} Pallas renderer on {platform}...")
    try:
        image, _ = render(gaussians, camera, mode=mode, use_pallas=True, backend=backend)
        jax.block_until_ready(image)
    except Exception as e:
        print(f"FAILED to run {mode} Pallas renderer: {e}")
        return None, None

    num_runs = 50
    start = time.perf_counter()
    for _ in range(num_runs):
        image, _ = render(gaussians, camera, mode=mode, use_pallas=True, backend=backend)
        jax.block_until_ready(image)
    end = time.perf_counter()
    t_render = (end - start) / num_runs

    # --- Training Step Benchmark ---
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    camera_static = (camera.W, camera.H, camera.fx, camera.fy, camera.cx, camera.cy)
    
    print(f"Warmup {mode} Pallas training step on {platform}...")
    try:
        state, loss, metrics = train_step(state, target, camera.W2C, camera_static, optimizer, 
                                         mode=mode, use_pallas=True, backend=backend)
        jax.block_until_ready(loss)
    except Exception as e:
        import traceback
        print(f"FAILED to run {mode} Pallas training: {e}")
        traceback.print_exc()
        return t_render, None

    start = time.perf_counter()
    for _ in range(num_runs):
        state, loss, metrics = train_step(state, target, camera.W2C, camera_static, optimizer, 
                                         mode=mode, use_pallas=True, backend=backend)
        jax.block_until_ready(loss)
    end = time.perf_counter()
    t_train = (end - start) / num_runs
    
    return t_render, t_train

def test_pallas_3dgs_vs_2dgs():
    print(f"\nJAX Devices: {jax.devices()}")
    xyz, rgb, jax_cameras, jax_targets = load_fern()
    
    print("\n--- 3DGS Pallas Benchmark ---")
    t_3dgs_render, t_3dgs_train = benchmark_pallas("3dgs", xyz, rgb, jax_cameras, jax_targets)
    
    print("\n--- 2DGS Pallas Benchmark ---")
    t_2dgs_render, t_2dgs_train = benchmark_pallas("2dgs", xyz, rgb, jax_cameras, jax_targets)
    
    results = {
        "3dgs": {"render": t_3dgs_render, "train": t_3dgs_train},
        "2dgs": {"render": t_2dgs_render, "train": t_2dgs_train}
    }
    
    print("\n--- Summary ---")
    if t_3dgs_render and t_2dgs_render:
        print(f"3DGS Render: {t_3dgs_render*1000:.2f} ms")
        print(f"2DGS Render: {t_2dgs_render*1000:.2f} ms")
        print(f"Renderer Slowdown: {t_2dgs_render/t_3dgs_render:.2f}x")
    
    if t_3dgs_train and t_2dgs_train:
        print(f"3DGS Train: {t_3dgs_train*1000:.2f} ms")
        print(f"2DGS Train: {t_2dgs_train*1000:.2f} ms")
        print(f"Training Slowdown: {t_2dgs_train/t_3dgs_train:.2f}x")
        
    # Write to file
    with open("BENCHMARK_PALLAS_2DGS_3DGS.md", "w") as f:
        f.write("# Pallas Benchmark: 3DGS vs 2DGS\n\n")
        f.write(f"**Platform:** {jax.devices()[0].platform}\n\n")
        f.write("| Mode | Render Latency | Training Latency |\n")
        f.write("| :--- | :--- | :--- |\n")
        if t_3dgs_render and t_3dgs_train:
            f.write(f"| 3DGS | {t_3dgs_render*1000:.2f} ms | {t_3dgs_train*1000:.2f} ms |\n")
        if t_2dgs_render and t_2dgs_train:
            f.write(f"| 2DGS | {t_2dgs_render*1000:.2f} ms | {t_2dgs_train*1000:.2f} ms |\n")
        
        f.write("\n## Comparison\n")
        if t_3dgs_render and t_2dgs_render:
            f.write(f"- Renderer Slowdown (2DGS/3DGS): {t_2dgs_render/t_3dgs_render:.2f}x\n")
        if t_3dgs_train and t_2dgs_train:
            f.write(f"- Training Slowdown (2DGS/3DGS): {t_2dgs_train/t_3dgs_train:.2f}x\n")

if __name__ == "__main__":
    test_pallas_3dgs_vs_2dgs()
