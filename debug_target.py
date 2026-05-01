import jax
import numpy as np
from jax_gs.io.colmap import load_colmap_dataset

path = "gs://dataset-nerf/nerf_llff_data/room"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")

print(f"Room Target min: {np.min(jax_targets[0])}, max: {np.max(jax_targets[0])}")

path = "gs://dataset-nerf/nerf_llff_data/fern"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")

print(f"Fern Target min: {np.min(jax_targets[0])}, max: {np.max(jax_targets[0])}")
