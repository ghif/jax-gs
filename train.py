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
import time
from tqdm import tqdm
from PIL import Image
from functools import partial
import concurrent.futures

from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.io.ply import save_ply
from jax_gs.training.trainer import train_step
from jax_gs.training.density import init_density_state, densify_and_prune, reset_opacities
from jax_gs.renderer.renderer import render

def save_artifacts_task(gaussians_dict, iteration, progress_dir, ply_path, camera, fast_tpu_rasterizer, render_fn, save_ply_fn, sh_degree):
    """Task to be run in a background thread."""
    from jax_gs.core.gaussians import Gaussians
    gaussians = Gaussians(**gaussians_dict)
    
    # Trigger render (async on TPU)
    img, _ = render_fn(gaussians, camera, fast_tpu_rasterizer=fast_tpu_rasterizer, sh_degree=sh_degree)
    
    # Materialize image to host (blocks thread, but not main training loop)
    img_np = np.array(img)
    
    # Save image
    img_path = f"{progress_dir}/progress_{iteration:04d}.png"
    with fsspec.open(img_path, "wb") as f:
        pil_img = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        f.write(buf.getvalue())
    
    # Save PLY (overwrite the same file to save space)
    save_ply_fn(ply_path, gaussians)

def get_active_gaussians(state):
    """Extracts only the active Gaussians into a host-side dictionary for saving."""
    # REVERT: Move contiguous arrays to host first. Slicing on device from host-loop is slow.
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
                 images_subdir: str = "images_8",
                 max_gaussians_cap: int = 200_000,
                 max_gaussians_growth: int = 8,
                 density_interval: int = 500):
    
    # Infer scene name from path
    scene_name = os.path.basename(data_path.rstrip('/'))
    print(f"Training on scene: {scene_name} (mode: 3dgs)")
    print(f"Fast TPU Rasterizer: {fast_tpu_rasterizer}")
    density_interval = max(1, density_interval)
    max_gaussians_growth = max(1, max_gaussians_growth)

    init_fn = init_gaussians_from_pcd

    @partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
    def train_block(state, rng_key, all_targets, all_w2cs, steps_per_block, camera_static, optimizer, fast_tpu_rasterizer, sh_degree):
        rng_key, subkey = jax.random.split(rng_key)
        idxs = jax.random.randint(subkey, (steps_per_block,), 0, all_targets.shape[0])
        batch_targets = all_targets[idxs]
        batch_w2cs = all_w2cs[idxs]

        def one_step(carry, inputs):
            state = carry
            target, w2c = inputs
            
            # Perform training step
            state, loss, metrics = train_step(state, target, w2c, camera_static, optimizer, fast_tpu_rasterizer=fast_tpu_rasterizer, sh_degree=sh_degree)
            
            return state, (loss, metrics)

        state, (losses, metrics) = jax.lax.scan(one_step, state, (batch_targets, batch_w2cs))
        avg_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
        avg_metrics["max_interactions_per_tile"] = jnp.max(metrics["max_interactions_per_tile"])
        avg_metrics["overflow_tiles"] = jnp.max(metrics["overflow_tiles"])
        avg_metrics["overflow_interactions"] = jnp.max(metrics["overflow_interactions"])
        avg_metrics["radius_cap_violations"] = jnp.max(metrics["radius_cap_violations"])
        return state, rng_key, losses, avg_metrics
        
    # 1. Load Data
    path = data_path
    xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, images_subdir)
    
    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(jax_cameras)} cameras for training")
    
    # Calculate scene extent for scaling learning rates and density thresholds
    xyz_mean = np.mean(xyz, axis=0)
    extent = np.max(np.linalg.norm(xyz - xyz_mean, axis=1))
    print(f"Scene extent: {extent:.4f}")
    
    # 2. Initialize Gaussians
    gaussians = init_fn(np.array(xyz), np.array(rgb))
    
    # 3. Setup Optimizer and DensityState
    # Parameter-specific learning rates (tuned for JAX pixel-averaged loss)
    # Using paper scaling (0.00016 * extent). Increased by 3x for faster convergence with JAX.
    means_lr_init = 0.00016 * extent * 3.0
    means_lr_end = 0.0000016 * extent * 3.0
    means_lr_schedule = optax.exponential_decay(
        init_value=means_lr_init,
        transition_steps=num_iterations,
        decay_rate=means_lr_end / means_lr_init
    )
    
    from jax_gs.core.gaussians import Gaussians
    param_labels = Gaussians(
        means="means",
        scales="scales",
        quaternions="quaternions",
        opacities="opacities",
        sh_coeffs="sh_coeffs"
    )
    
    optimizer = optax.multi_transform(
        {
            "means": optax.adam(learning_rate=means_lr_schedule),
            "scales": optax.adam(learning_rate=0.005),
            "quaternions": optax.adam(learning_rate=0.001),
            "opacities": optax.adam(learning_rate=0.05),
            "sh_coeffs": optax.adam(learning_rate=0.0025),
        },
        param_labels
    )
    
    print(f"Using 3DGS multi-parameter optimizer (Means LR: {means_lr_init:.2e} -> {means_lr_end:.2e})")
    
    # Keep padded state bounded: all projection, sorting, and optimizer work scales with this static size.
    max_gaussians = max(len(xyz), min(len(xyz) * max_gaussians_growth, max_gaussians_cap))
    print(
        f"Initializing DensityState with max_gaussians={max_gaussians} "
        f"(initial: {len(xyz)}, growth: {max_gaussians_growth}x, cap: {max_gaussians_cap})"
    )
    state = init_density_state(gaussians, optimizer, max_gaussians)
    
    @jax.jit
    def density_step(state, rng_key):
        # Using paper default (0.0002) as trainer.py already handles loss normalization compensation.
        return densify_and_prune(state, rng_key, extent=extent, grad_threshold=0.0002)
        
    @jax.jit
    def opacity_reset_step(state):
        return reset_opacities(state)
    
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
        # SH Degree Scheduling: Increase degree every 1000 iterations (approx)
        curr_iter = b * steps_per_block
        sh_degree = min(3, curr_iter // 1000)

        block_start = time.perf_counter()
        curr_state, curr_rng, losses, block_metrics = train_block(
            curr_state, curr_rng, all_targets, all_w2cs, 
            steps_per_block, camera_static, optimizer, fast_tpu_rasterizer, sh_degree
        )
        
        avg_loss = jnp.mean(losses)
        avg_loss, block_metrics = jax.block_until_ready((avg_loss, block_metrics))
        avg_loss = float(avg_loss)
        block_metrics = {k: float(v) for k, v in block_metrics.items()}
        block_time = time.perf_counter() - block_start
        curr_iter = (b + 1) * steps_per_block
        density_time = 0.0
        reset_time = 0.0
        
        # Adaptive Density Control
        if 500 < curr_iter <= 15000 and curr_iter % density_interval == 0:
            curr_rng, density_rng = jax.random.split(curr_rng)
            density_start = time.perf_counter()
            curr_state = density_step(curr_state, density_rng)
            jax.block_until_ready(curr_state.active_mask.sum())
            density_time = time.perf_counter() - density_start
            
            # Periodically reset opacities (every 3000 steps)
            if curr_iter % 3000 == 0:
                reset_start = time.perf_counter()
                curr_state = opacity_reset_step(curr_state)
                jax.block_until_ready(jnp.sum(curr_state.gaussians.opacities))
                reset_time = time.perf_counter() - reset_start
                
        num_active = int(jax.block_until_ready(curr_state.active_mask.sum()))
        it_per_sec = steps_per_block / block_time if block_time > 0 else 0.0
        pbar.set_description(
            f"Loss: {avg_loss:.4f} | Active: {num_active}/{max_gaussians} | "
            f"SH: {sh_degree} | {it_per_sec:.1f} it/s"
        )
        if b % 10 == 0 or density_time > 0.0 or reset_time > 0.0:
            print(
                f"Block {b}/{num_blocks} | Iter: {curr_iter} | Loss: {avg_loss:.4f} | "
                f"L1: {block_metrics['l1']:.4f} | SSIM: {block_metrics['ssim']:.4f} | "
                f"Active: {num_active}/{max_gaussians} | SH: {sh_degree} | "
                f"Train: {block_time:.3f}s ({it_per_sec:.1f} it/s) | "
                f"Density: {density_time:.3f}s | Reset: {reset_time:.3f}s | "
                f"Interactions avg/max: {block_metrics['mean_interactions_per_tile']:.1f}/"
                f"{block_metrics['max_interactions_per_tile']:.0f} | "
                f"Overflow tiles/interactions: {block_metrics['overflow_tiles']:.0f}/"
                f"{block_metrics['overflow_interactions']:.0f} | "
                f"Radius cap: {block_metrics['radius_cap_violations']:.0f}"
            )
        
        if curr_iter % 1000 == 0:
            # Capture state for background task
            snap_gaussians_dict = get_active_gaussians(curr_state)
            
            # Submit background task (inject render and save_ply functions)
            ply_path = f"{ply_dir}/{scene_name}_latest.ply"
            fut = executor.submit(
                save_artifacts_task, 
                snap_gaussians_dict, curr_iter, progress_dir, ply_path, 
                jax_cameras[0], fast_tpu_rasterizer, render, save_ply, sh_degree
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
    parser.add_argument("--max_gaussians_cap", type=int, default=200_000, help="Upper bound for padded Gaussian capacity")
    parser.add_argument("--max_gaussians_growth", type=int, default=8, help="Capacity multiplier applied to the initial COLMAP point count")
    parser.add_argument("--density_interval", type=int, default=500, help="Run densify/prune every N iterations during the density window")
    args = parser.parse_args()
    
    run_training(num_iterations=args.num_iterations, 
                 data_path=args.data_path, output_base=args.output_path,
                 fast_tpu_rasterizer=args.fast_tpu_rasterizer,
                 images_subdir=args.images_subdir,
                 max_gaussians_cap=args.max_gaussians_cap,
                 max_gaussians_growth=args.max_gaussians_growth,
                 density_interval=args.density_interval)
