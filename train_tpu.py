import jax
import jax.numpy as jnp
import optax
import numpy as np
import os
import datetime
import random
import argparse
from tqdm import tqdm
from PIL import Image
from functools import partial

from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.renderer.renderer import render
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.io.ply import save_ply
from jax_gs.training.trainer import train_step
from jax_gs.core.camera import Camera

def run_training_tpu(num_iterations: int = 10000):
    # 0. TPU Init
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Found {num_devices} devices: {devices}")
    
    # 1. Load Data
    path = "data/nerf_example_data/nerf_llff_data/fern"
    xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")
    
    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(jax_cameras)} cameras for training")
    
    # 2. Initialize Gaussians
    gaussians = init_gaussians_from_pcd(np.array(xyz), np.array(rgb))
    
    # 3. Setup Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    
    # 4. Multi-device setup
    # Replicate gaussians and opt_state across all devices
    replicated_gaussians = jax.device_put_replicated(gaussians, devices)
    replicated_opt_state = jax.device_put_replicated(opt_state, devices)
    
    # Parallel training step
    @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(4, 5))
    def p_train_step(params, opt_state, target_image, w2c, camera_static, optimizer):
        W, H, fx, fy, cx, cy = camera_static
        camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
        
        lambda_ssim = 0.2
        def loss_fn(p):
            image = render(p, camera)
            l1 = l1_loss(image, target_image)
            d_ssim = d_ssim_loss(image, target_image)
            return (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # ALL-REDUCE GRADIENTS: This is the key for TPU speedup with data parallelism
        grads = jax.lax.pmean(grads, axis_name='batch')
        
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        
        return next_params, next_opt_state, loss

    # 5. Training Loop
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"fern_tpu_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    progress_dir = os.path.join(output_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)

    pbar = tqdm(range(num_iterations // num_devices))
    
    params = replicated_gaussians
    opt_state = replicated_opt_state

    # Static imports for the loss functions
    from jax_gs.training.losses import l1_loss, d_ssim_loss

    for i in pbar:
        # Pick random cameras for each device
        idxs = [random.randint(0, len(jax_cameras)-1) for _ in range(num_devices)]
        batch_w2c = jnp.stack([jax_cameras[idx].W2C for idx in idxs])
        batch_targets = jnp.stack([jax_targets[idx] for idx in idxs])
        
        # Assume all cameras in this batch have same intrinsic (true for LLFF)
        cam0 = jax_cameras[idxs[0]]
        camera_static = (int(cam0.W), int(cam0.H), float(cam0.fx), float(cam0.fy), float(cam0.cx), float(cam0.cy))
        
        params, opt_state, losses = p_train_step(params, opt_state, batch_targets, batch_w2c, camera_static, optimizer)
        
        if i % 10 == 0:
            avg_loss = jnp.mean(losses)
            pbar.set_description(f"Loss: {avg_loss:.4f}")
            
            if i % 100 == 0:
                # Get params from device 0 for rendering
                single_gaussians = jax.tree_map(lambda x: x[0], params)
                img = render(single_gaussians, jax_cameras[0])
                img_np = np.array(img)
                Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(progress_dir, f"progress_{i*num_devices:04d}.png")
                )

    # Final Save
    print("Training done. Saving final model...")
    final_gaussians = jax.tree_map(lambda x: x[0], params)
    save_ply(os.path.join(output_dir, "fern_final.ply"), final_gaussians)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=10000)
    args = parser.parse_args()
    
    run_training_tpu(num_iterations=args.num_iterations)
