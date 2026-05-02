import jax
import jax.numpy as jnp
import optax
import numpy as np
import os
import datetime
import random
import argparse
import fsspec
import io
from tqdm import tqdm
from PIL import Image
from functools import partial
import concurrent.futures

from jax_gs.io.colmap import load_colmap_dataset

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(4, 5, 6, 7, 8, 9))
def train_block(state, rng_key, all_targets, all_w2cs, steps_per_block, camera_static, optimizer, mode, use_pallas, backend):
    if mode == "2dgs":
        from jax_2dgs.training.trainer import train_step_internal
    else:
        from jax_gs.training.trainer import train_step_internal
            
    def one_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        
        # Sample camera index
        idx = jax.random.randint(subkey, (), 0, all_targets.shape[0])
        target = all_targets[idx]
        w2c = all_w2cs[idx]
        
        # Perform training step
        state, loss, metrics = train_step_internal(state, target, w2c, camera_static, optimizer, use_pallas, backend)
        
        return (state, key), loss

    (state, rng_key), losses = jax.lax.scan(one_step, (state, rng_key), None, length=steps_per_block)
    return state, rng_key, losses

def save_artifacts_task(gaussians, iteration, progress_dir, ply_dir, camera, use_pallas, backend, mode):
    """Task to be run in a background thread."""
    if mode == "2dgs":
        from jax_2dgs.renderer.renderer import render
        from jax_2dgs.io.ply import save_ply_2d as save_ply
    else:
        from jax_gs.renderer.renderer import render
        from jax_gs.io.ply import save_ply
        
    # Trigger render (async on TPU)
    img, _ = render(gaussians, camera, use_pallas=use_pallas, backend=backend)
    
    # Materialize image to host (blocks thread, but not main training loop)
    img_np = np.array(img)
    
    # Save image
    img_path = f"{progress_dir}/progress_{iteration:04d}.png"
    with fsspec.open(img_path, "wb") as f:
        pil_img = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        f.write(buf.getvalue())
    
    # Save PLY (materializes Gaussians to host)
    save_ply(f"{ply_dir}/fern_splats_{iteration:04d}.ply", gaussians)

def run_parallel_training(num_iterations: int = 30000, mode: str = "3dgs", 
                          data_path: str = "gs://dataset-nerf/nerf_llff_data/fern",
                          output_base: str = "gs://dataset-nerf/results",
                          use_pallas: bool = False,
                          backend: str = "tpu"):
    
    # 0. TPU Init
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Found {num_devices} devices: {devices}")
    
    # Conditional imports based on mode
    if mode == "2dgs":
        from jax_2dgs.core.gaussians_2d import init_gaussians_2d_from_pcd
        from jax_2dgs.io.ply import save_ply_2d as save_ply
        from jax_2dgs.renderer.renderer import render
        init_fn = init_gaussians_2d_from_pcd
    else:
        from jax_gs.core.gaussians import init_gaussians_from_pcd
        from jax_gs.io.ply import save_ply
        from jax_gs.renderer.renderer import render
        init_fn = init_gaussians_from_pcd

    # 1. Load Data
    path = data_path
    xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")
    
    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(jax_cameras)} cameras for training")
    
    # 2. Initialize Gaussians
    gaussians = init_fn(np.array(xyz), np.array(rgb))
    
    # 3. Setup Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    
    # 4. Prepare data on device
    all_targets = jnp.stack(jax_targets)
    all_w2cs = jnp.stack([c.W2C for c in jax_cameras])
    
    mesh = jax.sharding.Mesh(devices, ('batch',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
    
    replicated_state = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.broadcast_to(x, (num_devices,) + x.shape), sharding), 
        state
    )
    replicated_targets = jax.device_put(jnp.broadcast_to(all_targets, (num_devices,) + all_targets.shape), sharding)
    replicated_w2cs = jax.device_put(jnp.broadcast_to(all_w2cs, (num_devices,) + all_w2cs.shape), sharding)
    
    rng = jax.random.PRNGKey(random.randint(0, 10000))
    rng_per_device = jax.random.split(rng, num_devices)
    rng_sharded = jax.device_put(rng_per_device, sharding)
    
    # 5. Training Loop
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_base}/fern_parallel_{mode}_{timestamp}"
    
    fs, _ = fsspec.core.url_to_fs(output_dir)
    if fs.protocol == 'file' or (isinstance(fs.protocol, (list, tuple)) and 'file' in fs.protocol):
        os.makedirs(output_dir, exist_ok=True)
    
    progress_dir = f"{output_dir}/progress"
    ply_dir = f"{output_dir}/ply"
    
    if fs.protocol == 'file' or (isinstance(fs.protocol, (list, tuple)) and 'file' in fs.protocol):
        os.makedirs(progress_dir, exist_ok=True)
        os.makedirs(ply_dir, exist_ok=True)

    steps_per_block = 500
    num_blocks = num_iterations // steps_per_block
    
    pbar = tqdm(range(num_blocks))
    
    cam0 = jax_cameras[0]
    camera_static = (int(cam0.W), int(cam0.H), float(cam0.fx), float(cam0.fy), float(cam0.cx), float(cam0.cy))
    
    curr_state = replicated_state
    curr_rng = rng_sharded

    # Background executor for I/O and rendering
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    futures = []

    for b in pbar:
        # Parallel training block
        curr_state, curr_rng, losses = train_block(
            curr_state, curr_rng, replicated_targets, replicated_w2cs, 
            steps_per_block, camera_static, optimizer, mode, use_pallas, backend
        )
        
        avg_loss = jnp.mean(losses[0])
        pbar.set_description(f"Loss: {avg_loss:.4f}")
        
        curr_iter = (b + 1) * steps_per_block
        
        if curr_iter % 1000 == 0:
            # Capture state for background task
            # Note: JAX arrays are immutable, so this is a reference to the state at this point.
            # Transfer to host will happen in the thread.
            snap_gaussians = jax.tree_util.tree_map(lambda x: x[0], curr_state[0])
            
            # Submit background task
            fut = executor.submit(
                save_artifacts_task, 
                snap_gaussians, curr_iter, progress_dir, ply_dir, 
                jax_cameras[0], use_pallas, backend, mode
            )
            futures.append(fut)
            
            # Keep only the last few futures to avoid memory pressure
            futures = [f for f in futures if not f.done()]

    # Final Save
    print("Waiting for background tasks to complete...")
    concurrent.futures.wait(futures)
    executor.shutdown()

    print("Training done. Saving final model...")
    final_gaussians = jax.tree_util.tree_map(lambda x: x[0], curr_state[0])
    save_ply(f"{output_dir}/fern_final.ply", final_gaussians)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=30000)
    parser.add_argument("--mode", type=str, default="3dgs", choices=["3dgs", "2dgs"])
    parser.add_argument("--data_path", type=str, default="gs://dataset-nerf/nerf_llff_data/fern")
    parser.add_argument("--output_path", type=str, default="gs://dataset-nerf/results")
    parser.add_argument("--use_pallas", action="store_true", help="Use Pallas kernels for rasterization")
    parser.add_argument("--backend", type=str, default="tpu", choices=["gpu", "tpu"])
    args = parser.parse_args()
    
    run_parallel_training(num_iterations=args.num_iterations, mode=args.mode, 
                          data_path=args.data_path, output_base=args.output_path,
                          use_pallas=args.use_pallas, backend=args.backend)
