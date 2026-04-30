import numpy as np
from jax_gs.io.colmap import load_colmap_dataset
path = "gs://dataset-nerf/nerf_llff_data/room"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")
print(f"XYZ min: {np.min(xyz, axis=0)}, max: {np.max(xyz, axis=0)}")
print(f"Num cameras: {len(jax_cameras)}")
for i, cam in enumerate(jax_cameras[:2]):
    print(f"Cam {i}: W={cam.W}, H={cam.H}, fx={cam.fx}, fy={cam.fy}, cx={cam.cx}, cy={cam.cy}")
    w2c = cam.W2C
    print(f"  W2C det: {np.linalg.det(w2c[:3,:3])}, T: {w2c[:3, 3]}")
