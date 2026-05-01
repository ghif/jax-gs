import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import os
from PIL import Image
from jax_gs.core.gaussians import init_gaussians_from_pcd, Gaussians
from jax_gs.core.camera import Camera
from jax_gs.renderer.renderer import render
from jax_gs.io.colmap import load_colmap_dataset

def test_visual_comparison():
    # Run Fern test
    print("\n--- Running Fern Dataset Comparison ---")
    xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset("data/nerf_example_data/nerf_llff_data/fern", "images_8")
    gaussians = init_gaussians_from_pcd(jnp.array(xyz), jnp.array(rgb))
    cam = jax_cameras[0]
    
    # Render JAX
    print("Rendering JAX (CPU)...")
    img_jax = np.array(render(gaussians, cam, use_mlx=False))
    
    # Render MLX
    print("Rendering MLX (GPU)...")
    img_mlx = np.array(render(gaussians, cam, use_mlx=True))
    
    # Debug: Print interaction summary if possible
    # We'd need to expose it from render() or call it manually
    
    os.makedirs("results/tests", exist_ok=True)
    Image.fromarray((img_jax * 255).astype(np.uint8)).save("results/tests/fern_jax_cpu.png")
    Image.fromarray((img_mlx * 255).astype(np.uint8)).save("results/tests/fern_mlx_gpu.png")
    
    diff = np.abs(img_jax - img_mlx)
    mean_diff = np.mean(diff)
    print(f"Fern Mean Diff: {mean_diff:.6f}")
    
    # Run Single Gaussian check
    print("\n--- Running Single Gaussian Sanity Check ---")
    p = jnp.array([[0.0, 0.0, 5.0]]) # Center
    c = jnp.array([[1.0, 0.0, 0.0]]) # Red
    # Create a reasonably sized Gaussian
    g_single = init_gaussians_from_pcd(p, c).replace(
        scales=jnp.array([[-1.0, -1.0, -1.0]]),
        opacities=jnp.array([[10.0]])
    )
    cam_s = Camera(W=128, H=128, fx=100.0, fy=100.0, cx=64.0, cy=64.0, W2C=jnp.eye(4), full_proj=jnp.eye(4))
    
    img_s_jax = render(g_single, cam_s, use_mlx=False)
    img_s_mlx = render(g_single, cam_s, use_mlx=True)
    
    img_s_jax_np = np.array(img_s_jax)
    img_s_mlx_np = np.array(img_s_mlx)
    
    Image.fromarray((img_s_jax_np * 255).astype(np.uint8)).save("results/tests/single_jax.png")
    Image.fromarray((img_s_mlx_np * 255).astype(np.uint8)).save("results/tests/single_mlx.png")
    
    diff_s = np.abs(img_s_jax_np - img_s_mlx_np)
    print(f"Single Gaussian Mean Diff: {np.mean(diff_s):.6f}")
    
    # Assert consistency
    assert mean_diff < 0.001, f"Fern Mean Diff too high: {mean_diff}"
    assert np.mean(diff_s) < 0.0001, f"Single Gaussian Mean Diff too high: {np.mean(diff_s)}"

if __name__ == "__main__":
    test_visual_comparison()
    print("\nTest PASSED: JAX and MLX are visually consistent.")
