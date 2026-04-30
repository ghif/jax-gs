import jax
import jax.numpy as jnp
def test_clip(x):
    y = x * 0.282 + 0.5
    y = jnp.clip(y, 0.0, 1.0)
    return jnp.sum(y)

x = jnp.array([-1.7724538509055159]) # corresponds to rgb=0.0
print(f"y evaluated: {x * 0.282 + 0.5}")
print(f"Gradient: {jax.grad(test_clip)(x)}")
