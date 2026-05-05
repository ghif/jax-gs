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
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.io.ply import save_ply
from jax_gs.training.trainer import train_step
from jax_gs.training.density import init_density_state, densify_and_prune
from jax_gs.renderer.renderer import render

def save_artifacts_task(gaussians_dict, iteration, progress_dir, ply_dir, camera, fast_tpu_rasterizer, scene_name, render_fn, save_ply_fn):
    """Task to be run in a background thread."""
    from jax_gs.core.gaussians import Gaussians
    gaussians = Gaussians(**gaussians_dict)
    
    # Trigger render (async on TPU)
    img, _ = render_fn(gaussians, camera, fast_tpu_rasterizer=fast_tpu_rasterizer)
    
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
    save_ply_fn(f"{ply_dir}/{scene_name}_splats_{iteration:04d}.ply", gaussians)

def get_active_gaussians(state):
    """Extracts only the active Gaussians into a host-side dictionary for saving."""
    active = np.array(state.active_mask)
    g = state.gaussians
    return {
        "means": np.array(g.means)[active],
        "scales": np.array(g.scales)[active],
        "quaternions": np.array(g.quaternions)[active],
        "opacities": np.array(g.opacities)[active],
        "sh_coeffs": np.array(g.sh_coeffs)[active]
    }

def run_training(num_iterations: int = 30000,
                 data_path: str = "gs://dataset-nerf/nerf_llff_data/fern",
                 output_base: str = "gs://dataset-nerf/results",
                 fast_tpu_rasterizer: bool = False,
                 images_subdir: str = "images_8"):
    
    # Infer scene name from path
    scene_name = os.path.basename(data_path.rstrip('/'))
    print(f"Training on scene: {scene_name} (mode: 3dgs)")
    print(f"Fast TPU Rasterizer: {fast_tpu_rasterizer}")

    init_fn = init_gaussians_from_pcd

    @partial(jax.jit, static_argnums=(4, 5, 6, 7))
    def train_block(state, rng_key, all_targets, all_w2cs, steps_per_block, camera_static, optimizer, fast_tpu_rasterizer):
        def one_step(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            
            # Sample camera index
            idx = jax.random.randint(subkey, (), 0, all_targets.shape[0])
            target = all_targets[idx]
            w2c = all_w2cs[idx]
            
            # Perform training step
            state, loss, metrics = train_step(state, target, w2c, camera_static, optimizer, fast_tpu_rasterizer=fast_tpu_rasterizer)
            
            return (state, key), loss

        (state, rng_key), losses = jax.lax.scan(one_step, (state, rng_key), None, length=steps_per_block)
        return state, rng_key, losses
        
    @jax.jit
    def density_step(state):
        return densify_and_prune(state, extent=5.0)

    # 1. Load Data
    path = data_path
    xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, images_subdir)
    
    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(jax_cameras)} cameras for training")
    
    # 2. Initialize Gaussians
    gaussians = init_fn(np.array(xyz), np.array(rgb))
    
    # 3. Setup Optimizer and DensityState
    optimizer = optax.adam(learning_rate=1e-3)
    max_gaussians = min(2_000_000, len(xyz) * 4) # Max buffer size
    print(f"Initializing DensityState with max_gaussians={max_gaussians}")
    state = init_density_state(gaussians, optimizer, max_gaussians)
    
    # 4. Prepare data on device
    all_targets = jnp.stack(jax_targets)
    all_w2cs = jnp.stack([c.W2C for c in jax_cameras])
    
    rng = jax.random.PRNGKey(random.randint(0, 10000))
    
    # 5. Training Loop
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rasterizer_suffix = "_fast_tpu" if fast_tpu_rasterizer else ""
    output_dir = f"{output_base}/{scene_name}_3dgs{rasterizer_suffix}_{timestamp}"
    
    fs, _ = fsspec.core.url_to_fs(output_dir)
    if fs.protocol == 'file' or (isinstance(fs.protocol, (list, tuple)) and 'file' in fs.protocol):
        os.makedirs(output_dir, exist_ok=True)
    
    progress_dir = f"{output_dir}/progress"
    ply_dir = f"{output_dir}/ply"
    
    if fs.protocol == 'file' or (isinstance(fs.protocol, (list, tuple)) and 'file' in fs.protocol):
        os.makedirs(progress_dir, exist_ok=True)
        os.makedirs(ply_dir, exist_ok=True)

    # We use blocks of 100 to allow frequent density control
    steps_per_block = 100
    num_blocks = num_iterations // steps_per_block
    
    pbar = tqdm(range(num_blocks))
    
    cam0 = jax_cameras[0]
    camera_static = (int(cam0.W), int(cam0.H), float(cam0.fx), float(cam0.fy), float(cam0.cx), float(cam0.cy))
    
    curr_state = state
    curr_rng = rng

    # Background executor for I/O and rendering
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    futures = []

    for b in pbar:
        # Compiled training block
        curr_state, curr_rng, losses = train_block(
            curr_state, curr_rng, all_targets, all_w2cs, 
            steps_per_block, camera_static, optimizer, fast_tpu_rasterizer
        )
        
        avg_loss = jnp.mean(losses)
        curr_iter = (b + 1) * steps_per_block
        
        # Adaptive Density Control
        if 500 < curr_iter <= 15000:
            curr_state = density_step(curr_state)
            num_active = curr_state.active_mask.sum().item()
            pbar.set_description(f"Loss: {avg_loss:.4f} | Active: {num_active}")
        else:
            num_active = curr_state.active_mask.sum().item()
            pbar.set_description(f"Loss: {avg_loss:.4f} | Active: {num_active}")
        
        if curr_iter % 1000 == 0:
            # Capture state for background task
            snap_gaussians_dict = get_active_gaussians(curr_state)
            
            # Submit background task (inject render and save_ply functions)
            fut = executor.submit(
                save_artifacts_task, 
                snap_gaussians_dict, curr_iter, progress_dir, ply_dir, 
                jax_cameras[0], fast_tpu_rasterizer, scene_name, render, save_ply
            )
            futures.append(fut)
            
            # Keep only the last few futures to avoid memory pressure
            futures = [f for f in futures if not f.done()]

    # Final Save
    print("Waiting for background tasks to complete...")
    concurrent.futures.wait(futures)
    executor.shutdown()

    print("Training done. Saving final model...")
    from jax_gs.core.gaussians import Gaussians
    final_gaussians = Gaussians(**get_active_gaussians(curr_state))
    save_ply(f"{output_dir}/{scene_name}_final.ply", final_gaussians)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=30000)
    parser.add_argument("--data_path", type=str, default="gs://dataset-nerf/nerf_llff_data/fern")
    parser.add_argument("--output_path", type=str, default="gs://dataset-nerf/results")
    parser.add_argument("--images_subdir", type=str, default="images_8")
    parser.add_argument("--fast_tpu_rasterizer", action="store_true", help="Use the optimized JAX scan rasterizer for TPU")
    args = parser.parse_args()
    
    run_training(num_iterations=args.num_iterations, 
                 data_path=args.data_path, output_base=args.output_path,
                 fast_tpu_rasterizer=args.fast_tpu_rasterizer,
                 images_subdir=args.images_subdir)

