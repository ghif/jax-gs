import jax
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

from jax_gs.io.colmap import load_colmap_dataset
import jax.numpy as jnp

def run_training(num_iterations: int = 10000, mode: str = "3dgs", 
                 data_path: str = "gs://dataset-nerf/tandt/truck",
                 output_base: str = "./results",
                 use_pallas: bool = False,
                 backend: str = "gpu"):
    
    # Conditional imports based on mode
    if mode == "2dgs":
        from jax_2dgs.core.gaussians_2d import init_gaussians_2d_from_pcd
        from jax_2dgs.training.trainer import train_step, train_step_parallel
        from jax_2dgs.renderer.renderer import render
        from jax_2dgs.io.ply import save_ply_2d as save_ply
        init_fn = init_gaussians_2d_from_pcd
    else:
        from jax_gs.core.gaussians import init_gaussians_from_pcd
        from jax_gs.training.trainer import train_step, train_step_parallel
        from jax_gs.renderer.renderer import render
        from jax_gs.io.ply import save_ply
        init_fn = init_gaussians_from_pcd

    # 1. Load Data
    path = data_path
    xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images")
    
    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(jax_cameras)} cameras for training")
    
    # 2. Initialize Gaussians
    gaussians = init_fn(np.array(xyz), np.array(rgb))
    
    # 3. Setup Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    
    # Data Parallelism Setup
    num_devices = jax.local_device_count()
    if num_devices > 1:
        print(f"Using Data Parallelism across {num_devices} devices")
        sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.local_devices(), 'batch'), jax.sharding.PartitionSpec('batch'))
        state = jax.tree_util.tree_map(
            lambda x: jax.device_put(jnp.broadcast_to(x, (num_devices,) + x.shape), sharding), 
            state
        )
    else:
        print("Using single device training")

    # 4. Training Loop
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_base}/truck_{mode}_{timestamp}"
    
    # Use fsspec to handle directory creation if local
    fs, _ = fsspec.core.url_to_fs(output_dir)
    if fs.protocol == 'file' or (isinstance(fs.protocol, (list, tuple)) and 'file' in fs.protocol):
        os.makedirs(output_dir, exist_ok=True)
    
    progress_dir = f"{output_dir}/progress"
    ply_dir = f"{output_dir}/ply"
    
    if fs.protocol == 'file' or (isinstance(fs.protocol, (list, tuple)) and 'file' in fs.protocol):
        os.makedirs(progress_dir, exist_ok=True)
        os.makedirs(ply_dir, exist_ok=True)

    pbar = tqdm(range(num_iterations))
    
    for i in pbar:
        if num_devices > 1:
            # Batch multiple cameras
            indices = [random.randint(0, len(jax_cameras)-1) for _ in range(num_devices)]
            batch_cameras = [jax_cameras[idx] for idx in indices]
            batch_targets = jnp.stack([jax_targets[idx] for idx in indices])
            batch_w2c = jnp.stack([cam.W2C for cam in batch_cameras])
            
            # Assume all cameras in batch have same static params (resolution/intrinsics)
            cam = batch_cameras[0]
            camera_static = (int(cam.W), int(cam.H), float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy))
            
            state, loss, metrics = train_step_parallel(state, batch_targets, batch_w2c, camera_static, optimizer, 
                                                     use_pallas, backend)
            # Loss and metrics are replicated, take the first one
            loss = loss[0]
        else:
            # Pick random camera
            idx = random.randint(0, len(jax_cameras)-1)
            cam = jax_cameras[idx]
            target = jax_targets[idx]
            
            camera_static = (int(cam.W), int(cam.H), float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy))
            
            state, loss, metrics = train_step(state, target, cam.W2C, camera_static, optimizer, 
                                             use_pallas=use_pallas, backend=backend)
        
        if i % 10 == 0:
            pbar.set_description(f"Loss: {loss:.4f}")
            
            if i % 100 == 0:
                # Render logic (unreplicate for rendering/saving)
                curr_gaussians = state[0]
                if num_devices > 1:
                    curr_gaussians = jax.tree_util.tree_map(lambda x: x[0], curr_gaussians)

                img, _ = render(curr_gaussians, jax_cameras[0], use_pallas=use_pallas, backend=backend)
                img_np = np.array(img)
                
                # Save image via fsspec
                img_path = f"{progress_dir}/progress_{i:04d}.png"
                with fsspec.open(img_path, "wb") as f:
                    pil_img = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    f.write(buf.getvalue())
                
                save_ply(f"{ply_dir}/truck_splats_{i:04d}.ply", curr_gaussians) 

    # Final Save
    print("Training done. Saving final model...")
    final_gaussians = state[0]
    if num_devices > 1:
        final_gaussians = jax.tree_util.tree_map(lambda x: x[0], final_gaussians)

    save_ply(f"{output_dir}/truck_final.ply", final_gaussians)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--mode", type=str, default="3dgs", choices=["3dgs", "2dgs"])
    parser.add_argument("--data_path", type=str, default="gs://dataset-nerf/tandt/truck")
    parser.add_argument("--output_path", type=str, default="gs://dataset-nerf/results")
    parser.add_argument("--use_pallas", action="store_true", help="Use Pallas kernels for rasterization")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "tpu"])
    args = parser.parse_args()
    
    run_training(num_iterations=args.num_iterations, mode=args.mode, 
                 data_path=args.data_path, output_base=args.output_path,
                 use_pallas=args.use_pallas, backend=args.backend)
