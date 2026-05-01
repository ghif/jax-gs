import jax
import optax
import numpy as np
from jax_gs.io.colmap import load_colmap_dataset
from jax_gs.core.gaussians_2d import init_gaussians_2d_from_pcd
from jax_gs.renderer.renderer import render
from jax_gs.training.trainer import train_step

path = "gs://dataset-nerf/nerf_llff_data/room"
xyz, rgb, jax_cameras, jax_targets = load_colmap_dataset(path, "images_8")

# Fix RGB if they are all black
# if np.max(rgb) == 0.0:
#     rgb = np.full_like(rgb, 0.5)

gaussians = init_gaussians_2d_from_pcd(np.array(xyz), np.array(rgb))
cam = jax_cameras[0]
target = jax_targets[0]

optimizer = optax.adam(learning_rate=1e-2) # higher LR
opt_state = optimizer.init(gaussians)
state = (gaussians, opt_state)

camera_static = (int(cam.W), int(cam.H), float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy))

for i in range(101):
    state, loss, metrics = train_step(state, target, cam.W2C, camera_static, optimizer, use_pallas=False, mode="2dgs", backend="gpu")
    if i % 20 == 0:
        g = state[0]
        c = g.sh_coeffs[:, 0, :] * 0.282 + 0.5
        print(f"Step {i}: Loss {loss:.4f}, Color max {np.max(c):.4f}")

img, _ = render(state[0], cam, mode="2dgs")
print(f"Final img max: {np.max(np.array(img))}")
import jax.numpy as jnp
from jax_gs.training.trainer import loss_fn

grad_fn = jax.grad(lambda p: loss_fn(p, target, cam.W2C, camera_static, False, "2dgs", "gpu")[0])
grads = grad_fn(gaussians)
print(f"SH grads max: {np.max(np.abs(grads.sh_coeffs))}")
print(f"Opacities grads max: {np.max(np.abs(grads.opacities))}")
print(f"Means grads max: {np.max(np.abs(grads.means))}")
