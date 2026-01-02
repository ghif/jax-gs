
import jax
# Enable/Disable x64 based on env
# jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
import time
import os

# Import renderers
from renderer_v2 import render_v2 as render_cpu, Camera
# For MPS, we need to handle imports carefully if x64 is forced off
try:
    from renderer_v2_mps import render_v2_mps
    HAS_MPS = True
except ImportError:
    HAS_MPS = False
    print("MPS Renderer not found or import failed.")

from gaussians import init_gaussians_from_pcd

def benchmark():
    print("Initializing Benchmark Data...")
    # Create dummy data
    N = 100000 # 100k points
    points = np.random.rand(N, 3).astype(np.float32) * 5.0 - 2.5
    colors = np.random.rand(N, 3).astype(np.float32)
    
    # Init Gaussians (CPU/default)
    gaussians = init_gaussians_from_pcd(jnp.array(points), jnp.array(colors))
    
    # Create Camera
    W, H = 800, 600
    fx, fy = 800.0, 800.0
    cx, cy = 400.0, 300.0
    w2c = jnp.eye(4)
    camera = Camera(W, H, fx, fy, cx, cy, w2c, jnp.eye(4))
    
    print(f"Benchmarking with {N} Gaussians, Image {W}x{H}")

    # --- CPU Benchmark ---
    print("\n--- Benchmarking CPU Renderer (renderer_v2) ---")
    try:
        # JIT compile
        start = time.time()
        render_jitter = jax.jit(render_cpu, static_argnums=(1,)) # Camera is tuple, maybe static? No, Camera is NamedTuple.
        # Actually Camera has static fields W, H. JIT might need handling.
        # Let's use the decomposed one for easier JIT if needed, or just JIT the wrapper if it handles NamedTuple (JAX usually does).
        # But wait, renderer_v2 signature: (gaussians, camera, background).
        # Camera is a NamedTuple. NamedTuples are registered as pytrees.
        # However, W and H are integers. If they are used in reshape/loop bounds, they MUST be static.
        # renderer_v2 uses camera.W/H for reshaping.
        
        # We need to wrap it to make W/H static or assume they are static in NamedTuple?
        # JAX treats NamedTuple leaves as tracers. integers in NamedTuple are treated as JAX arrays (weakly typed) or tracers.
        # We likely need to re-wrap or use the decomposed function for benchmarking to be safe.
        
        from renderer_v2 import render_camera_v2
        
        # (gaussians, W2C, fx, fy, cx, cy, W, H, background)
        # We need to partial apply W and H
        render_cpu_jit = jax.jit(render_camera_v2, static_argnums=(6, 7), backend='cpu')
        
        # Warmup
        print("Warmup (CPU)...")
        _ = render_cpu_jit(gaussians, w2c, fx, fy, cx, cy, W, H, jnp.array([0.,0.,0.]))
        jax.block_until_ready(_)
        print("Warmup done.")
        
        t0 = time.time()
        for _ in range(10):
            res = render_cpu_jit(gaussians, w2c, fx, fy, cx, cy, W, H, jnp.array([0.,0.,0.]))
            jax.block_until_ready(res)
        avg_cpu = (time.time() - t0) / 10.0
        print(f"CPU Average Time: {avg_cpu*1000:.2f} ms")
        
    except Exception as e:
        print(f"CPU Benchmark Failed: {e}")

    # --- MPS Benchmark ---
    if HAS_MPS:
        print("\n--- Benchmarking MPS Renderer (renderer_v2_mps) ---")
        try:
            from renderer_v2_mps import render_camera_v2_mps
            import jax.numpy as jnp_mps
            
            # Cast inputs to float32 explicitly as enforced in MPS script
            g_means = gaussians.means.astype(jnp.float32)
            g_scales = gaussians.scales.astype(jnp.float32)
            g_quats = gaussians.quaternions.astype(jnp.float32)
            g_opacities = gaussians.opacities.astype(jnp.float32)
            g_sh = gaussians.sh_coeffs.astype(jnp.float32)
            
            from gaussians import Gaussians
            g_mps = Gaussians(
                means=g_means, 
                scales=g_scales, 
                quaternions=g_quats, 
                opacities=g_opacities, 
                sh_coeffs=g_sh
            )
            
            w2c_f32 = w2c.astype(jnp.float32)
            # Scalars as floats
            fx_f = float(fx)
            fy_f = float(fy)
            cx_f = float(cx)
            cy_f = float(cy)
            
            render_mps_jit = jax.jit(render_camera_v2_mps, static_argnums=(6, 7))
            
            # Warmup
            print("Warmup (MPS)...")
            bg = jnp.zeros(3, dtype=jnp.float32)
            _ = render_mps_jit(g_mps, w2c_f32, fx_f, fy_f, cx_f, cy_f, W, H, bg)
            jax.block_until_ready(_)
            print("Warmup done.")
            
            t0 = time.time()
            for _ in range(10):
                res = render_mps_jit(g_mps, w2c_f32, fx_f, fy_f, cx_f, cy_f, W, H, bg)
                jax.block_until_ready(res)
            avg_mps = (time.time() - t0) / 10.0
            print(f"MPS Average Time: {avg_mps*1000:.2f} ms")
            
            if avg_cpu > 0:
                print(f"\nSpeedup: {avg_cpu / avg_mps:.2f}x")
            
        except Exception as e:
            print(f"MPS Benchmark Failed: {e}")

if __name__ == "__main__":
    benchmark()
