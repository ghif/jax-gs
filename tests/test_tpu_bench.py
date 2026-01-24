import jax
import jax.numpy as jnp
import time
import numpy as np
from jax_gs.core.gaussians import Gaussians
from jax_gs.renderer.renderer import render
from jax_gs.core.camera import Camera
from functools import partial

def benchmark_tpu():
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Benchmarking on {num_devices} devices: {devices}")
    
    # Initialize some dummy splats
    num_points = 100_000
    gaussians = Gaussians(
        means=jnp.zeros((num_points, 3)),
        sh_coeffs=jnp.zeros((num_points, 1, 3)),
        opacities=jnp.zeros((num_points, 1)),
        scales=jnp.zeros((num_points, 3)),
        quaternions=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (num_points, 1))
    )
    
    # Initialize a dummy camera
    camera = Camera(
        W=800, H=800, fx=500, fy=500, cx=400, cy=400,
        W2C=jnp.eye(4), full_proj=jnp.eye(4)
    )
    
    # Replicate across devices
    gaussians_repl = jax.device_put_replicated(gaussians, devices)
    
    @partial(jax.pmap, in_axes=(0, None, None, None, None, None, None, None), 
                     static_broadcasted_argnums=(2, 3, 4, 5, 6, 7))
    def p_render(g, w2c, W, H, fx, fy, cx, cy):
        c = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
        return render(g, c)

    # Warmup
    print("Warmup...")
    p_render(gaussians_repl, camera.W2C, camera.W, camera.H, camera.fx, camera.fy, camera.cx, camera.cy)
    
    # Run benchmark
    num_iters = 50
    print(f"Running {num_iters} iterations...")
    start_time = time.time()
    for _ in range(num_iters):
        res = p_render(gaussians_repl, camera.W2C, camera.W, camera.H, camera.fx, camera.fy, camera.cx, camera.cy)
        res.block_until_ready()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = (num_iters * num_devices) / total_time
    
    print(f"Total time for {num_iters} iterations on {num_devices} cores: {total_time:.4f}s")
    print(f"Throughput (Total FPS): {fps:.2f} frames/s")
    print(f"Per-core latency: {total_time/num_iters:.4f}s")

if __name__ == "__main__":
    benchmark_tpu()
