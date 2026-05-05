import jax
import jax.numpy as jnp
import time

@jax.jit
def compact_indices(mask):
    return jnp.argsort((~mask).astype(jnp.int8))

mask = jax.random.bernoulli(jax.random.PRNGKey(0), 0.1, (2000000,))
indices = compact_indices(mask)
jax.block_until_ready(indices) # Warmup

start = time.perf_counter()
indices = compact_indices(mask)
jax.block_until_ready(indices)
print("Time:", time.perf_counter() - start)
