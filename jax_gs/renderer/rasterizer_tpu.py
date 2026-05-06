import jax
import jax.numpy as jnp
from jax_gs.renderer.rasterizer import TILE_SIZE, BLOCK_SIZE, MAX_TILE_CHUNKS, MAX_TILE_INTERACTIONS

def _invert_covariance_2d(cov2D):
    """Inverts 2D covariance matrices for Gaussian exponent calculation."""
    det = jnp.maximum(cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2, 1e-6)
    return jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)

def render_tiles_tpu(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, background):
    """
    Highly optimized TPU rasterization using fully vectorized XLA primitives and jax.checkpoint.
    
    Architectural Differences vs Standard Rasterizer:
    1.  **Full Vectorization**: Instead of nested vmaps (tiles -> pixels), this version flattens
        tiles and pixels into a single dimension [num_tiles, 256]. This allows XLA to better
        utilize the TPU's MXU by creating larger matrix operations.
    2.  **Memory Efficiency**: Uses `jax.checkpoint` on the main `scan` loop. This forces JAX
        to recompute intermediate values during the backward pass instead of storing them all,
        which is critical for preventing OOM when training with high resolutions or many Gaussians.
    3.  **Broadcasted Gather**: Prefetches all Gaussian data for every tile into a massive
        tensor [num_tiles, BLOCK_SIZE, D]. This avoids repeated random-access memory lookups
        inside the tight inner loop.

    Args:
        means2D, cov2D, opacities, colors: Gaussian parameters.
        sorted_tile_ids, sorted_gaussian_ids: Sorted interactions from get_tile_interactions.
        H, W: Target image resolution.
        background: RGB background color.
    """
    num_tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
    num_tiles = num_tiles_x * num_tiles_y
    
    inv_cov2D = _invert_covariance_2d(cov2D)
    sig_opacities = jax.nn.sigmoid(opacities)
    
    # Identify the start and end of each tile's Gaussian list in the global sorted array
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, jnp.arange(num_tiles + 1))
    valid_ids = jnp.where(sorted_gaussian_ids < means2D.shape[0], sorted_gaussian_ids, 0)
    
    tile_starts = tile_boundaries[:-1]
    tile_ends = tile_boundaries[1:]
    tile_counts = tile_ends - tile_starts # Number of real Gaussians per tile
    
    tile_counts = jnp.minimum(tile_counts, MAX_TILE_INTERACTIONS)
    
    # Pre-calculate global pixel coordinates for EVERY pixel in the image, 
    # grouped by tile. Shape: [num_tiles, 256] (since 16x16 = 256)
    idx = jnp.arange(256, dtype=jnp.int32)
    tx = jnp.arange(num_tiles) % num_tiles_x
    ty = jnp.arange(num_tiles) // num_tiles_x
    
    px = (tx[:, None] * TILE_SIZE).astype(jnp.float32) + (idx % TILE_SIZE)[None, :].astype(jnp.float32) + 0.5
    py = (ty[:, None] * TILE_SIZE).astype(jnp.float32) + (idx // TILE_SIZE)[None, :].astype(jnp.float32) + 0.5
    
    @jax.checkpoint
    def scan_fn(carry, inputs):
        """
        The core alpha-blending loop.
        Processes the i-th Gaussian for ALL pixels in ALL tiles simultaneously.
        """
        c_accum, T = carry
        mu_x, mu_y, ic00, ic01, ic11, op, cols, is_active_local = inputs
        mu_x = mu_x[:, None]
        mu_y = mu_y[:, None]
        ic00 = ic00[:, None]
        ic01 = ic01[:, None]
        ic11 = ic11[:, None]
        op = op[:, None]
        is_active_local = is_active_local[:, None]

        dx = px - mu_x
        dy = py - mu_y
        power = -0.5 * (dx * dx * ic00 + 2.0 * dx * dy * ic01 + dy * dy * ic11)
        power = jnp.where(is_active_local, power, -100.0)
        alpha = jnp.exp(power) * op
        is_active = is_active_local & (T > 1e-4)
        alpha = jnp.where((power > -10.0) & is_active, jnp.minimum(0.99, alpha), 0.0)

        weight = alpha * T
        c_accum = c_accum + weight[:, :, None] * cols[:, None, :]
        T = T * (1.0 - alpha)

        return (c_accum, T), None

    def chunk_scan(carry, chunk_idx):
        def do_chunk(chunk_carry):
            c_accum, T = chunk_carry
            chunk_start = tile_starts[:, None] + chunk_idx * BLOCK_SIZE + jnp.arange(BLOCK_SIZE)[None, :]
            max_idx = jnp.maximum(valid_ids.shape[0] - 1, 0)
            gather_indices = jnp.clip(chunk_start, 0, max_idx)
            local_mask = (chunk_idx * BLOCK_SIZE + jnp.arange(BLOCK_SIZE)[None, :]) < tile_counts[:, None]
            tile_gids = valid_ids[gather_indices]

            g_means = means2D[tile_gids]
            g_icov = inv_cov2D.reshape(-1, 4)[tile_gids]
            g_ops = sig_opacities[tile_gids]
            g_cols = colors[tile_gids]

            scan_inputs = (
                g_means[:, :, 0].T,
                g_means[:, :, 1].T,
                g_icov[:, :, 0].T,
                g_icov[:, :, 1].T,
                g_icov[:, :, 3].T,
                g_ops[:, :, 0].T,
                jnp.swapaxes(g_cols, 0, 1),
                local_mask.T,
            )
            return jax.lax.scan(scan_fn, (c_accum, T), scan_inputs)[0]

        return jax.lax.cond(chunk_idx < required_chunks, do_chunk, lambda x: x, carry), None
        
    # Initialize accumulation buffers
    c_init = jnp.zeros((num_tiles, 256, 3), dtype=jnp.float32)
    T_init = jnp.ones((num_tiles, 256), dtype=jnp.float32)
    
    required_chunks = jnp.maximum(
        1,
        jnp.minimum(MAX_TILE_CHUNKS, (jnp.max(tile_counts) + BLOCK_SIZE - 1) // BLOCK_SIZE)
    )
    (final_c, final_T), _ = jax.lax.scan(chunk_scan, (c_init, T_init), jnp.arange(MAX_TILE_CHUNKS))
    
    # Add the background color based on remaining transmittance
    final_color = final_c + final_T[:, :, None] * background[None, None, :]
    
    # Reshape the flat tile-pixel dimension [num_tiles, 256] back into an image [H, W, 3]
    out_grid = final_color.reshape(num_tiles_y, num_tiles_x, TILE_SIZE, TILE_SIZE, 3)
    out_image = out_grid.swapaxes(1, 2).reshape(num_tiles_y * TILE_SIZE, num_tiles_x * TILE_SIZE, 3)
    
    return out_image[:H, :W, :]
