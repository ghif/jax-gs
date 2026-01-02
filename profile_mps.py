
import jax
import jax.numpy as jnp
import time
import sys

# Import components to test
from renderer_v2_mps import project_gaussians, get_tile_interactions, render_tiles, Camera
from gaussians import Gaussians, init_gaussians_from_pcd

def profile():
    print("Profiling MPS Components...")
    
    # 1. Setup Data
    N = 1000
    points = jnp.zeros((N, 3), dtype=jnp.float32)
    colors = jnp.zeros((N, 3), dtype=jnp.float32)
    
    scales = jnp.full((N, 3), -3.0, dtype=jnp.float32)
    quats = jnp.tile(jnp.array([1.,0.,0.,0.], dtype=jnp.float32), (N, 1))
    opacities = jnp.zeros((N, 1), dtype=jnp.float32)
    sh = jnp.zeros((N, 16, 3), dtype=jnp.float32)
    
    gaussians = Gaussians(
        means=points, 
        scales=scales, 
        quaternions=quats, 
        opacities=opacities, 
        sh_coeffs=sh
    )
    
    W, H = 800, 600
    w2c = jnp.eye(4, dtype=jnp.float32)
    camera = Camera(W, H, 800., 800., 400., 300., w2c, jnp.eye(4, dtype=jnp.float32))

    # --- Profile Projection ---
    print("Profiling Projection...")
    @jax.jit
    def proj_fn(g, c):
        return project_gaussians(g, c)
    
    # Warmup
    _ = proj_fn(gaussians, camera)
    jax.block_until_ready(_)
    
    t0 = time.time()
    for _ in range(10):
        res = proj_fn(gaussians, camera)
        jax.block_until_ready(res)
    print(f"Projection Time: {(time.time()-t0)*100:.2f} ms")
    
    # Get outputs for next stage
    means2D, cov2D, radii, valid_mask, depths = proj_fn(gaussians, camera)
    
    # --- Profile Sorting ---
    print("Profiling Sorting (get_tile_interactions)...")
    @jax.jit
    def sort_fn(m2d, r, vm, d):
        return get_tile_interactions(m2d, r, vm, d, H, W, 16)
        
    # Warmup
    _ = sort_fn(means2D, radii, valid_mask, depths)
    jax.block_until_ready(_)
    
    t0 = time.time()
    for _ in range(10):
        res = sort_fn(means2D, radii, valid_mask, depths)
        jax.block_until_ready(res)
    print(f"Sorting Time: {(time.time()-t0)*100:.2f} ms")
    
    tid, gid, count = sort_fn(means2D, radii, valid_mask, depths)
    
    # --- Profile Rasterization ---
    print("Profiling Rasterization...")
    # Mock colors
    colors = jnp.zeros((N, 3), dtype=jnp.float32)
    bg = jnp.zeros(3, dtype=jnp.float32)
    
    @jax.jit
    def rast_fn(m2d, c2d, op, cols, dep, stid, sgid):
        return render_tiles(m2d, c2d, op, cols, dep, stid, sgid, H, W, 16, bg)
    
    # Warmup
    _ = rast_fn(means2D, cov2D, opacities, colors, depths, tid, gid)
    jax.block_until_ready(_)
    
    t0 = time.time()
    for _ in range(10):
        res = rast_fn(means2D, cov2D, opacities, colors, depths, tid, gid)
        jax.block_until_ready(res)
    print(f"Rasterization Time: {(time.time()-t0)*100:.2f} ms")

if __name__ == "__main__":
    profile()
