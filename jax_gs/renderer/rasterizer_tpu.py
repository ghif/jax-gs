import jax
import jax.numpy as jnp
from jax_gs.renderer.rasterizer import TILE_SIZE, BLOCK_SIZE

def _invert_covariance_2d(cov2D):
    det = jnp.maximum(cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2, 1e-6)
    return jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)

def render_tiles_tpu(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background):
    """
    Highly optimized TPU rasterization using fully vectorized XLA primitives and jax.checkpoint.
    This replaces the brittle Mosaic Pallas kernel with a robust, faster implementation.
    """
    num_tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
    num_tiles = num_tiles_x * num_tiles_y
    
    inv_cov2D = _invert_covariance_2d(cov2D)
    sig_opacities = jax.nn.sigmoid(opacities)
    
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, jnp.arange(num_tiles + 1))
    valid_ids = jnp.where(sorted_gaussian_ids < means2D.shape[0], sorted_gaussian_ids, 0)
    
    # Calculate counts to enforce local mask
    tile_starts = tile_boundaries[:-1]
    tile_ends = tile_boundaries[1:]
    tile_counts = tile_ends - tile_starts # [num_tiles]
    
    all_tile_indices = tile_starts[:, None] + jnp.arange(BLOCK_SIZE)[None, :]
    max_idx = jnp.maximum(valid_ids.shape[0] - 1, 0)
    all_tile_indices = jnp.clip(all_tile_indices, 0, max_idx)
    
    # Create valid mask
    local_mask = jnp.arange(BLOCK_SIZE)[None, :] < tile_counts[:, None] # [num_tiles, BLOCK_SIZE]
    
    tile_gids = valid_ids[all_tile_indices]
    
    g_means = means2D[tile_gids]
    g_icov = inv_cov2D.reshape(-1, 4)[tile_gids]
    g_ops = sig_opacities[tile_gids]
    g_cols = colors[tile_gids]
    
    idx = jnp.arange(256, dtype=jnp.int32)
    tx = jnp.arange(num_tiles) % num_tiles_x
    ty = jnp.arange(num_tiles) // num_tiles_x
    
    px = (tx[:, None] * TILE_SIZE).astype(jnp.float32) + (idx % TILE_SIZE)[None, :].astype(jnp.float32) + 0.5
    py = (ty[:, None] * TILE_SIZE).astype(jnp.float32) + (idx // TILE_SIZE)[None, :].astype(jnp.float32) + 0.5
    
    @jax.checkpoint
    def scan_fn(carry, i):
        c_accum, T = carry
        
        # Calculate alpha on the fly
        mu_x = g_means[:, i, 0][:, None]
        mu_y = g_means[:, i, 1][:, None]
        ic00 = g_icov[:, i, 0][:, None]
        ic01 = g_icov[:, i, 1][:, None]
        ic11 = g_icov[:, i, 3][:, None]
        op = g_ops[:, i, 0][:, None]
        
        dx = px - mu_x
        dy = py - mu_y
        
        power = -0.5 * (dx * dx * ic00 + 2.0 * dx * dy * ic01 + dy * dy * ic11)
        alpha = jnp.where(power > -10.0, jnp.exp(jnp.clip(power, -100.0, 0.0)) * op, 0.0)
        
        # Apply local mask
        is_active = local_mask[:, i][:, None] & (T > 1e-4)
        alpha = jnp.where(is_active, jnp.minimum(0.99, alpha), 0.0)
        
        weight = alpha * T
        c_accum = c_accum + weight[:, :, None] * g_cols[:, i, :][:, None, :]
        T = T * (1.0 - alpha)
        return (c_accum, T), None
        
    c_init = jnp.zeros((num_tiles, 256, 3), dtype=jnp.float32)
    T_init = jnp.ones((num_tiles, 256), dtype=jnp.float32)
    
    (final_c, final_T), _ = jax.lax.scan(scan_fn, (c_init, T_init), jnp.arange(BLOCK_SIZE))
    
    final_color = final_c + final_T[:, :, None] * background[None, None, :]
    
    out_grid = final_color.reshape(num_tiles_y, num_tiles_x, TILE_SIZE, TILE_SIZE, 3)
    out_image = out_grid.swapaxes(1, 2).reshape(num_tiles_y * TILE_SIZE, num_tiles_x * TILE_SIZE, 3)
    
    return out_image[:H, :W, :]
