import jax
import jax.numpy as jnp
import numpy as np
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.core.gaussians_2d import init_gaussians_2d_from_pcd
from jax_gs.renderer.projection_2d import project_gaussians_2d
from jax_gs.renderer.rasterizer import get_tile_interactions
from jax_gs.renderer.rasterizer_2d import render_tiles_2d

path = "gs://dataset-nerf/nerf_llff_data/room"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")

gaussians = init_gaussians_2d_from_pcd(np.array(xyz), np.array(rgb))
cam = jax_cameras[0]

means2D, cov2D, radii, valid_mask, depths, normals = project_gaussians_2d(gaussians, cam)

sorted_tile_ids, sorted_gaussian_ids, n_interactions = get_tile_interactions(
    means2D, radii, valid_mask, depths, cam.H, cam.W, 16
)

colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5

img, r_depth, r_depth_sq, r_normal, r_accum = render_tiles_2d(
    means2D, cov2D, gaussians.opacities, colors, depths, normals,
    sorted_tile_ids, sorted_gaussian_ids,
    cam.H, cam.W, 16, jnp.array([0.5, 0.5, 0.5])
)

img_np = np.array(img)
accum_np = np.array(r_accum)

print(f"Img min: {np.min(img_np)}, max: {np.max(img_np)}")
print(f"Accum min: {np.min(accum_np)}, max: {np.max(accum_np)}")
