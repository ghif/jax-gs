import jax
import jax.numpy as jnp
import optax
from jax_gs.training.losses import l1_loss, mse_loss
from jax_gs.training.trainer import train_step
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.camera import Camera

def test_losses():
    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.1, 1.9, 3.0])
    
    # L1: avg(0.1, 0.1, 0) = 0.0666...
    np_l1 = l1_loss(pred, target)
    assert jnp.allclose(np_l1, 0.06666667)
    
    # MSE: avg(0.01, 0.01, 0) = 0.00666...
    np_mse = mse_loss(pred, target)
    assert jnp.allclose(np_mse, 0.00666667)

def test_ssim_losses():
    # Setup mock images (H, W, C)
    H, W = 32, 32
    target = jax.random.uniform(jax.random.PRNGKey(0), (H, W, 3))
    pred = target + 0.1 * jax.random.uniform(jax.random.PRNGKey(1), (H, W, 3))
    
    from jax_gs.training.losses import ssim, d_ssim_loss
    
    val_ssim = ssim(pred, target)
    val_d_ssim = d_ssim_loss(pred, target)
    
    assert 0 <= val_ssim <= 1.0
    assert 0 <= val_d_ssim <= 1.0
    assert jnp.allclose(val_d_ssim, (1.0 - val_ssim) / 2.0)
    
    # Perfect match
    assert jnp.allclose(ssim(target, target), 1.0)
    assert jnp.allclose(d_ssim_loss(target, target), 0.0)

def test_train_step_execution():
    # Setup minimal training state
    num_points = 10
    means = jnp.zeros((num_points, 3))
    colors = jnp.zeros((num_points, 3))
    gaussians = init_gaussians_from_pcd(means, colors)
    
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    state = (gaussians, opt_state)
    
    # Mock data
    target_image = jnp.zeros((16, 16, 3))
    w2c = jnp.eye(4)
    camera_static = (16, 16, 10.0, 10.0, 8.0, 8.0) # W, H, fx, fy, cx, cy
    
    # Run step
    new_state, loss = train_step(state, target_image, w2c, camera_static, optimizer)
    
    assert loss >= 0
    assert len(new_state) == 2
    # Verify parameters changed (gradient flow)
    # Since target is 0 and init is 0 (sigmoid(op) > 0), loss will be > 0 and params should update
    assert not jnp.allclose(new_state[0].means, gaussians.means)
