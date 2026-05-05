import jax
import optax
from functools import partial
from jax_gs.renderer.renderer import render
from jax_gs.training.losses import l1_loss, d_ssim_loss
from jax_gs.core.camera import Camera
import jax.numpy as jnp
from jax_gs.training.density import DensityState
from jax_gs.renderer.projection import project_gaussians

def _compute_loss_and_metrics(params, target_image, w2c, camera_static, fast_tpu_rasterizer, active_mask, sh_degree=0):
    """
    Common loss computation logic for 3DGS.
    """
    W, H, fx, fy, cx, cy = camera_static
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    lambda_ssim = 0.2
    
    # We pass the active mask to the renderer so it ignores inactive padded Gaussians.
    image, extras = render(params, camera, fast_tpu_rasterizer=fast_tpu_rasterizer, mask=active_mask, sh_degree=sh_degree)
    
    l1 = l1_loss(image, target_image)
    d_ssim = d_ssim_loss(image, target_image)

    total_loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim

    metrics = {
        "l1": l1,
        "ssim": 1.0 - d_ssim * 2.0
    }

    return total_loss, metrics

def _mask_updates(updates, active_mask):
    def apply_mask(u):
        if u is None:
            return None
        # Add necessary axes to active_mask to broadcast to u.shape
        mask = active_mask
        for _ in range(u.ndim - 1):
            mask = mask[..., None]
        return jnp.where(mask, u, 0.0)
    return jax.tree_util.tree_map(apply_mask, updates)

def train_step_internal(state: DensityState, target_image, w2c, camera_static, optimizer, fast_tpu_rasterizer=False, sh_degree=0):
    """
    Internal training step for 3DGS, suitable for use inside scan/pmap.
    """
    params = state.gaussians
    opt_state = state.opt_state
    active = state.active_mask
    
    def loss_fn(p):
        return _compute_loss_and_metrics(p, target_image, w2c, camera_static, fast_tpu_rasterizer, active, sh_degree=sh_degree)
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # SPMD Average
    grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), grads)
    loss = jax.lax.pmean(loss, axis_name='batch')
    metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), metrics)
    
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    
    # Mask updates so we don't modify inactive padded Gaussians
    updates = _mask_updates(updates, active)
    next_params = optax.apply_updates(params, updates)
    
    # --- Exact 2D Gradients for Densification ---
    W, H, fx, fy, cx, cy = camera_static
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    means3D_homo = jnp.concatenate([params.means, jnp.ones((params.means.shape[0], 1))], axis=-1)
    means_cam = (means3D_homo @ camera.W2C.T)[:, :3]
    z = jnp.maximum(means_cam[:, 2], 0.01)
    
    # Approximation of 2D view-space positional gradient magnitude
    focal = (fx + fy) / 2.0
    grad_means3D = grads.means
    grad_norm_3d = jnp.linalg.norm(grad_means3D, axis=-1)
    
    # Scale by W * H to compensate for jnp.mean() loss normalization.
    # This makes the gradient magnitude consistent with sum-over-pixels expected by paper.
    grad_2d_mag = grad_norm_3d * (z / focal) * (float(W) * float(H))
    
    # Average across batch/devices
    grad_2d_mag = jax.lax.pmean(grad_2d_mag, axis_name='batch')
    
    # Update accumulators
    next_grad_accum = state.grad_accum + grad_2d_mag
    next_denom = state.denom + jnp.where(active, 1, 0)
    
    # Update max radii (we can just compute current radii)
    _, _, radii, _, _ = project_gaussians(params, camera, mask=active)
    radii = jax.lax.pmax(radii, axis_name='batch') # max across views
    next_max_radii = jnp.maximum(state.max_radii, radii)
    
    next_state = state.replace(
        gaussians=next_params,
        opt_state=next_opt_state,
        grad_accum=next_grad_accum,
        denom=next_denom,
        max_radii=next_max_radii
    )
    
    return next_state, loss, metrics

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def train_step(state: DensityState, target_image, w2c, camera_static, optimizer, fast_tpu_rasterizer=False, sh_degree=0):
    """
    Standard single-device training step for 3DGS.
    """
    params = state.gaussians
    opt_state = state.opt_state
    active = state.active_mask
    
    def loss_fn(p):
        return _compute_loss_and_metrics(p, target_image, w2c, camera_static, fast_tpu_rasterizer, active, sh_degree=sh_degree)
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    updates = _mask_updates(updates, active)
    next_params = optax.apply_updates(params, updates)
    
    W, H, fx, fy, cx, cy = camera_static
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    means3D_homo = jnp.concatenate([params.means, jnp.ones((params.means.shape[0], 1))], axis=-1)
    means_cam = (means3D_homo @ camera.W2C.T)[:, :3]
    z = jnp.maximum(means_cam[:, 2], 0.01)
    focal = (fx + fy) / 2.0
    # Scale by W * H to compensate for jnp.mean() loss normalization.
    grad_2d_mag = jnp.linalg.norm(grads.means, axis=-1) * (z / focal) * (float(W) * float(H))
    
    next_grad_accum = state.grad_accum + grad_2d_mag
    next_denom = state.denom + jnp.where(active, 1, 0)
    
    _, _, radii, _, _ = project_gaussians(params, camera, mask=active)
    next_max_radii = jnp.maximum(state.max_radii, radii)
    
    next_state = state.replace(
        gaussians=next_params,
        opt_state=next_opt_state,
        grad_accum=next_grad_accum,
        denom=next_denom,
        max_radii=next_max_radii
    )
    
    return next_state, loss, metrics
