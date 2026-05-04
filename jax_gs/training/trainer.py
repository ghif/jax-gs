import jax
import optax
from functools import partial
from jax_gs.renderer.renderer import render
from jax_gs.training.losses import l1_loss, d_ssim_loss
from jax_gs.core.camera import Camera
import jax.numpy as jnp

def _compute_loss_and_metrics(params, target_image, w2c, camera_static, fast_tpu_rasterizer):
    """
    Common loss computation logic for 3DGS.
    """
    W, H, fx, fy, cx, cy = camera_static
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    lambda_ssim = 0.2
    
    image, extras = render(params, camera, fast_tpu_rasterizer=fast_tpu_rasterizer)
    l1 = l1_loss(image, target_image)
    d_ssim = d_ssim_loss(image, target_image)

    total_loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim

    metrics = {
        "l1": l1,
        "ssim": 1.0 - d_ssim * 2.0
    }

    return total_loss, metrics

def train_step_internal(state, target_image, w2c, camera_static, optimizer, fast_tpu_rasterizer=False):
    """
    Internal training step for 3DGS, suitable for use inside scan/pmap.
    Expects to be called within a pmap with axis_name='batch'.
    """
    params, opt_state = state
    
    def loss_fn(p):
        return _compute_loss_and_metrics(p, target_image, w2c, camera_static, fast_tpu_rasterizer)
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Average gradients across all devices (SPMD)
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Average loss and metrics for logging consistency
    loss = jax.lax.pmean(loss, axis_name='batch')
    metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), metrics)
    
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    
    return (next_params, next_opt_state), loss, metrics

@partial(jax.jit, static_argnums=(3, 4, 5))
def train_step(state, target_image, w2c, camera_static, optimizer, fast_tpu_rasterizer=False):
    """
    Standard single-device training step for 3DGS.
    """
    params, opt_state = state
    
    def loss_fn(p):
        return _compute_loss_and_metrics(p, target_image, w2c, camera_static, fast_tpu_rasterizer)
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    
    return (next_params, next_opt_state), loss, metrics
