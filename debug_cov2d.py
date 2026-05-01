import jax
import numpy as np
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.core.gaussians_2d import init_gaussians_2d_from_pcd
from jax_gs.renderer.projection_2d import project_gaussians_2d

path = "gs://dataset-nerf/nerf_llff_data/room"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")

gaussians = init_gaussians_2d_from_pcd(np.array(xyz), np.array(rgb))
cam = jax_cameras[0]

means2D, cov2D, radii, valid_mask, depths, normals = project_gaussians_2d(gaussians, cam)

print(f"cov2D min: {np.min(cov2D[valid_mask])}, max: {np.max(cov2D[valid_mask])}")
print(f"det min: {np.min(cov2D[valid_mask, 0, 0] * cov2D[valid_mask, 1, 1] - cov2D[valid_mask, 0, 1]**2)}")

