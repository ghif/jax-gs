import jax
import numpy as np
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.core.gaussians_2d import init_gaussians_2d_from_pcd
from jax_gs.renderer.projection_2d import project_gaussians_2d
from jax_gs.renderer.rasterizer import get_tile_interactions

path = "gs://dataset-nerf/nerf_llff_data/room"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")

gaussians = init_gaussians_2d_from_pcd(np.array(xyz), np.array(rgb))
cam = jax_cameras[0]

means2D, cov2D, radii, valid_mask, depths, normals = project_gaussians_2d(gaussians, cam)

sorted_tile_ids, sorted_gaussian_ids, n_interactions = get_tile_interactions(
    means2D, radii, valid_mask, depths, cam.H, cam.W, 16
)

print(f"Number of interactions: {n_interactions}")
print(f"Sorted tile IDs min: {np.min(sorted_tile_ids)}, max: {np.max(sorted_tile_ids)}")
print(f"Max valid ID: {np.max(sorted_gaussian_ids[:n_interactions])}")

