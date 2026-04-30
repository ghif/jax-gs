import jax
import jax.numpy as jnp

def simple_render(c):
    weight = jnp.array([0.5, 0.5])
    c0 = c[0] * weight[0] + c[1] * weight[1]
    return jnp.sum(jnp.abs(c0 - 1.0))

print(jax.grad(simple_render)(jnp.array([0.0, 0.0])))
