import jax.numpy as jnp

def l1_loss(pred, target):
    """
    Mean Absolute Error.
    """
    return jnp.mean(jnp.abs(pred - target))

def mse_loss(pred, target):
    """
    Mean Squared Error.
    """
    return jnp.mean((pred - target) ** 2)
