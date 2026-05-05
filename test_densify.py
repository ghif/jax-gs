import jax
import jax.numpy as jnp

MAX_GAUSSIANS = 10

active = jnp.array([True, True, True, False, False, False, False, False, False, False])
clone_mask = jnp.array([True, False, False, False, False, False, False, False, False, False])
split_mask = jnp.array([False, True, False, False, False, False, False, False, False, False])
prune_mask = jnp.array([False, False, True, False, False, False, False, False, False, False])

active = active & ~prune_mask

num_clones = jnp.sum(clone_mask)
num_splits = jnp.sum(split_mask)
total_new = num_clones + num_splits

will_be_empty = ~active
empty_indices = jnp.argsort((~will_be_empty).astype(jnp.int8))
clone_src_indices = jnp.argsort((~clone_mask).astype(jnp.int8))
split_src_indices = jnp.argsort((~split_mask).astype(jnp.int8))

idx = jnp.arange(MAX_GAUSSIANS)
is_clone_dest = idx < num_clones
is_split_dest = (idx >= num_clones) & (idx < num_clones + num_splits)

src_to_read = jnp.where(is_clone_dest, clone_src_indices[idx], 0)
src_to_read = jnp.where(is_split_dest, split_src_indices[idx - num_clones], src_to_read)

new_values = jnp.where(is_clone_dest | is_split_dest, src_to_read, empty_indices)

gather_indices = jnp.arange(MAX_GAUSSIANS)
gather_indices = gather_indices.at[empty_indices].set(new_values)

new_split_mask = jnp.zeros(MAX_GAUSSIANS, dtype=bool)
new_split_mask = new_split_mask.at[empty_indices].set(is_split_dest)

new_active_mask = jnp.zeros(MAX_GAUSSIANS, dtype=bool)
new_active_mask = new_active_mask.at[empty_indices].set(is_clone_dest | is_split_dest)
final_active = active | new_active_mask

print("Empty Indices:", empty_indices)
print("Gather Indices:", gather_indices)
print("New Split Mask:", new_split_mask)
print("Final Active:", final_active)
