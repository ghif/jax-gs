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
