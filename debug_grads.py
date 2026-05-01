import jax
import jax.numpy as jnp
import numpy as np
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.core.gaussians_2d import init_gaussians_2d_from_pcd
from jax_gs.renderer.renderer import render

path = "gs://dataset-nerf/nerf_llff_data/room"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")

gaussians = init_gaussians_2d_from_pcd(np.array(xyz), np.array(rgb))
cam = jax_cameras[0]

def custom_loss(p):
    image, extras = render(p, cam, use_pallas=False, mode="2dgs", backend="gpu")
    return jnp.sum(image)

grad_fn = jax.grad(custom_loss)
grads = grad_fn(gaussians)

print(f"SH grads max: {np.max(np.abs(grads.sh_coeffs))}")
