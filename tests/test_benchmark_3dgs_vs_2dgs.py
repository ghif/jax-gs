import jax
import jax.numpy as jnp
import optax
import time
import os
import pytest
import numpy as np
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_2dgs.core.gaussians_2d import init_gaussians_2d_from_pcd
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.renderer.renderer import render as render_3d
from jax_2dgs.renderer.renderer import render as render_2d
from jax_gs.training.trainer import train_step

def load_fern():
    import fsspec
    path = "gs://dataset-nerf/nerf_llff_data/fern"
    fs, _ = fsspec.core.url_to_fs(path)
    if not fs.exists(path):
        pytest.skip("Fern dataset not found at expected path.")
    return load_colmap_dataset(path, "images_8")

def benchmark_renderer(mode: str, xyz, rgb, jax_cameras):
    camera = jax_cameras[0]
    if mode == "2dgs":
        gaussians = init_gaussians_2d_from_pcd(jnp.array(xyz), jnp.array(rgb))
        render_fn = render_2d
    else:
        gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
        render_fn = render_3d
    
    # Warmup
    print(f"Warmup {mode} renderer...")
    image, _ = render_fn(gaussians, camera)
    jax.block_until_ready(image)
    
    # Benchmark
    num_runs = 50
    start = time.perf_counter()
    for _ in range(num_runs):
        image, _ = render_fn(gaussians, camera)
        jax.block_until_ready(image)
    end = time.perf_counter()
    
    avg_time = (end - start) / num_runs
    return avg_time

def benchmark_training(mode: str, xyz, rgb, jax_cameras, jax_targets):
    camera = jax_cameras[0]
    target = jax_targets[0]
    if mode == "2dgs":
        from jax_2dgs.training.trainer import train_step as train_step_fn
        gaussians = init_gaussians_2d_from_pcd(jnp.array(xyz), jnp.array(rgb))
    else:
        from jax_gs.training.trainer import train_step as train_step_fn
        gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
        
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    camera_static = (camera.W, camera.H, camera.fx, camera.fy, camera.cx, camera.cy)
    
    # Warmup
    print(f"Warmup {mode} training step...")
    state, loss, metrics = train_step_fn(state, target, camera.W2C, camera_static, optimizer)
    jax.block_until_ready(loss)
    
    # Benchmark
    num_runs = 50
    start = time.perf_counter()
    for _ in range(num_runs):
        state, loss, metrics = train_step_fn(state, target, camera.W2C, camera_static, optimizer)
        jax.block_until_ready(loss)
    end = time.perf_counter()
    
    avg_time = (end - start) / num_runs
    return avg_time

def test_benchmark_3dgs_vs_2dgs():
    print(f"\nJAX Devices: {jax.devices()}")
    xyz, rgb, jax_cameras, jax_targets = load_fern()
    
    print("\n--- Renderer Benchmark ---")
    t_3dgs_render = benchmark_renderer("3dgs", xyz, rgb, jax_cameras)
    print(f"3DGS Renderer: {t_3dgs_render*1000:.2f} ms ({1/t_3dgs_render:.2f} FPS)")
    
    t_2dgs_render = benchmark_renderer("2dgs", xyz, rgb, jax_cameras)
    print(f"2DGS Renderer: {t_2dgs_render*1000:.2f} ms ({1/t_2dgs_render:.2f} FPS)")
    
    print("\n--- Training Step Benchmark ---")
    t_3dgs_train = benchmark_training("3dgs", xyz, rgb, jax_cameras, jax_targets)
    print(f"3DGS Training: {t_3dgs_train*1000:.2f} ms ({1/t_3dgs_train:.2f} it/s)")
    
    t_2dgs_train = benchmark_training("2dgs", xyz, rgb, jax_cameras, jax_targets)
    print(f"2DGS Training: {t_2dgs_train*1000:.2f} ms ({1/t_2dgs_train:.2f} it/s)")
    
    print("\n--- Summary ---")
    render_ratio = t_2dgs_render / t_3dgs_render
    train_ratio = t_2dgs_train / t_3dgs_train
    print(f"Renderer Slowdown (2DGS/3DGS): {render_ratio:.2f}x")
    print(f"Training Slowdown (2DGS/3DGS): {train_ratio:.2f}x")

if __name__ == "__main__":
    test_benchmark_3dgs_vs_2dgs()
