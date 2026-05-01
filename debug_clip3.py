import jax
import jax.numpy as jnp

def test_clip(x):
    y = jnp.clip(x, 0.0, 1.0)
    return jnp.sum(y)

print(jax.grad(test_clip)(-1e-7))
