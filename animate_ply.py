
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
import math

from jax_gs.io.ply import load_ply
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.renderer.renderer import render
from jax_gs.core.camera import Camera

def get_orbit_camera(center, distance, azimuth, elevation, W, H, fx, fy, cx, cy):
    """
    Constructs a Camera object orbiting a center point.
    """
    az = math.radians(azimuth)
    el = math.radians(elevation)
    
    # Position in world coords (Orbit around Y axis)
    # Swapped X and Z as requested to fix "front view"
    z = center[2] + distance * math.cos(el) * math.sin(az)
    y = center[1] + distance * math.sin(el)
    x = center[0] + distance * math.cos(el) * math.cos(az)
    
    pos = np.array([x, y, z])
    
    # Construct W2C
    forward = center - pos
    forward = forward / np.linalg.norm(forward)
    
    up = np.array([0, -1, 0]) 
    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-6)
    
    actual_up = np.cross(forward, right)
    
    R = np.stack([right, actual_up, forward], axis=0)
    t = -R @ pos
    
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    
    return Camera(
        W=int(W), H=int(H), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        W2C=jnp.array(w2c), full_proj=jnp.eye(4)
    )

def get_wiggle_camera(base_cam, azimuth_deg, elevation_deg, shift_scale=1.0):
    """
    Perturbs a base camera pose with a small wiggle.
    """
    w2c = np.array(base_cam.W2C)
    R_base = w2c[:3, :3]
    t_base = w2c[:3, 3]
    pos_base = -R_base.T @ t_base
    
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    
    s_az, c_az = math.sin(az), math.cos(az)
    s_el, c_el = math.sin(el), math.cos(el)
    
    # Rotation in camera local space
    R_wiggle_y = np.array([[c_az, 0, s_az], [0, 1, 0], [-s_az, 0, c_az]])
    R_wiggle_x = np.array([[1, 0, 0], [0, c_el, -s_el], [0, s_el, c_el]])
    R_wiggle = R_wiggle_y @ R_wiggle_x
    
    R_new = R_wiggle @ R_base
    
    # Shift along local right/up
    dx = s_az * shift_scale
    dy = s_el * shift_scale
    pos_new = pos_base + R_base.T @ np.array([dx, dy, 0])
    
    w2c_new = np.eye(4)
    w2c_new[:3, :3] = R_new
    w2c_new[:3, 3] = -R_new @ pos_new
    
    return Camera(
        W=base_cam.W, H=base_cam.H, fx=base_cam.fx, fy=base_cam.fy, cx=base_cam.cx, cy=base_cam.cy,
        W2C=jnp.array(w2c_new), full_proj=jnp.eye(4)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_path", type=str, help="Path to input PLY file")
    parser.add_argument("--output", type=str, default="results/room_parallel_3dgs_anim.mp4", help="Output MP4 path")
    parser.add_argument("--data_path", type=str, help="Optional COLMAP data path to load reference camera")
    parser.add_argument("--num_frames", type=int, default=150, help="Number of frames for one orbit")
    parser.add_argument("--fast_tpu", action="store_true", help="Use TPU optimized rasterizer")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree (0-3)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    args = parser.parse_args()
    
    print(f"Loading Gaussians from {args.ply_path}...")
    gaussians = load_ply(args.ply_path)
    
    W, H = args.width, args.height
    
    if args.data_path:
        print(f"Loading reference camera from {args.data_path}...")
        _, _, jax_cameras, _ = load_colmap_dataset(args.data_path, "images_8")
        base_cam = jax_cameras[0]
        if args.width != base_cam.W or args.height != base_cam.H:
            scale = args.width / base_cam.W
            base_cam = Camera(
                W=int(args.width), H=int(args.height),
                fx=float(base_cam.fx * scale), fy=float(base_cam.fy * scale),
                cx=float(base_cam.cx * scale), cy=float(base_cam.cy * scale),
                W2C=base_cam.W2C, full_proj=jnp.eye(4)
            )
    else:
        means_np = np.array(gaussians.means)
        valid_means_mask = ~np.any(np.isnan(means_np), axis=1)
        means_np = means_np[valid_means_mask]
        opacities_np = np.array(gaussians.opacities)[valid_means_mask].flatten()
        weights = 1.0 / (1.0 + np.exp(-np.clip(opacities_np, -20, 20)))
        center = np.average(means_np[weights > 0.5], axis=0, weights=weights[weights > 0.5]) if np.sum(weights > 0.5) > 100 else np.mean(means_np, axis=0)
        orbit_dist = np.percentile(np.linalg.norm(means_np - center, axis=1), 90) * 2.0
        fx = fy = W / (2 * math.tan(math.radians(60) / 2))
        cx, cy = W / 2, H / 2
        base_cam = get_orbit_camera(center, orbit_dist, 0, 0, W, H, fx, fy, cx, cy)

    temp_dir = "results/anim_frames_tmp"
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Initializing renderer...")
    _, _ = render(gaussians, base_cam, fast_tpu_rasterizer=args.fast_tpu, sh_degree=args.sh_degree)
    
    for i in tqdm(range(args.num_frames)):
        t = 2 * math.pi * i / args.num_frames
        az = 10.0 * math.sin(t) 
        el = 5.0 * math.cos(t)
        cam = get_wiggle_camera(base_cam, az, el, shift_scale=2.0)
        img, _ = render(gaussians, cam, fast_tpu_rasterizer=args.fast_tpu, sh_degree=args.sh_degree)
        Image.fromarray((np.clip(np.array(img), 0, 1) * 255).astype(np.uint8)).save(os.path.join(temp_dir, f"frame_{i:04d}.png"))
        
    print(f"Encoding video to {args.output}...")
    os.system(f"ffmpeg -framerate 30 -i {temp_dir}/frame_%04d.png -c:v libx264 -crf 20 -pix_fmt yuv420p -y {args.output}")
    
    # Cleanup
    for i in range(args.num_frames):
        try: os.remove(os.path.join(temp_dir, f"frame_{i:04d}.png"))
        except: pass
    try: os.rmdir(temp_dir)
    except: pass

if __name__ == "__main__":
    main()
