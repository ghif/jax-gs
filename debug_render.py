import jax
import numpy as np
from PIL import Image
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.core.gaussians_2d import init_gaussians_2d_from_pcd
from jax_gs.renderer.renderer import render

path = "gs://dataset-nerf/nerf_llff_data/fern"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")

gaussians = init_gaussians_2d_from_pcd(np.array(xyz), np.array(rgb))

img, extras = render(gaussians, jax_cameras[0], mode="2dgs", use_pallas=True, backend="tpu")
img_np = np.array(img)

print(f"Image shape: {img_np.shape}")
print(f"Image min: {np.min(img_np)}, max: {np.max(img_np)}")
print(f"NaN count: {np.sum(np.isnan(img_np))}")

if np.sum(np.isnan(img_np)) > 0:
    print("Found NaNs!")

