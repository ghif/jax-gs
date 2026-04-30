import numpy as np
from jax_gs.io.colmap import load_colmap_dataset
path = "gs://dataset-nerf/nerf_llff_data/fern"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")
print(f"XYZ min: {np.min(xyz, axis=0)}, max: {np.max(xyz, axis=0)}")
