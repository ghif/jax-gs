import jax
import jax.numpy as jnp

def test_fn(colors):
    def scan_fn(carry, i):
        weight = 0.5
        new_color = carry + weight * colors[i]
        return new_color, None
    res, _ = jax.lax.scan(scan_fn, jnp.zeros(3), jnp.arange(2))
    return jnp.sum(res)

colors = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
print(jax.grad(test_fn)(colors))
