import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
from jax_gs.core.gaussians import Gaussians
from jax_gs.core.camera import Camera
from jax_gs.renderer.renderer import render

def test_benchmark_tpu():
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
    sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(devices, 'batch'), jax.sharding.PartitionSpec('batch'))
    gaussians_repl = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.broadcast_to(x, (num_devices,) + x.shape), sharding),
        gaussians
    )

    @partial(jax.pmap, in_axes=(0, None, None, None, None, None, None, None),
                     static_broadcasted_argnums=(2, 3, 4, 5, 6, 7))
    def p_render(g, w2c, W, H, fx, fy, cx, cy):
        c = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
        return render(g, c)

    # Warmup
    print("Warmup...")
    res = p_render(gaussians_repl, camera.W2C, camera.W, camera.H, camera.fx, camera.fy, camera.cx, camera.cy)
    jax.block_until_ready(res)

    # Run benchmark
    num_iters = 50
    print(f"Running {num_iters} iterations...")
    start_time = time.time()
    for _ in range(num_iters):
        res = p_render(gaussians_repl, camera.W2C, camera.W, camera.H, camera.fx, camera.fy, camera.cx, camera.cy)
        jax.block_until_ready(res)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iters
    print(f"Average time per render: {avg_time * 1000:.2f} ms")
    print(f"FPS: {1/avg_time:.2f}")

if __name__ == "__main__":
    test_benchmark_tpu()
