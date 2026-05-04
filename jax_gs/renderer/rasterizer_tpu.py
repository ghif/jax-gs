import jax
import jax.numpy as jnp
from jax_gs.renderer.rasterizer import TILE_SIZE, BLOCK_SIZE

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
    
    # BROADCASTED GATHER: Construct indices for all Gaussians across all tiles.
    # Resulting shape: [num_tiles, BLOCK_SIZE]
    all_tile_indices = tile_starts[:, None] + jnp.arange(BLOCK_SIZE)[None, :]
    max_idx = jnp.maximum(valid_ids.shape[0] - 1, 0)
    all_tile_indices = jnp.clip(all_tile_indices, 0, max_idx)
    
    # Create a mask to distinguish real Gaussians from padding inside each block
    local_mask = jnp.arange(BLOCK_SIZE)[None, :] < tile_counts[:, None] # [num_tiles, BLOCK_SIZE]
    
    # Prefetch Gaussian data for all tiles at once
    tile_gids = valid_ids[all_tile_indices] # [num_tiles, BLOCK_SIZE]
    
    g_means = means2D[tile_gids]            # [num_tiles, BLOCK_SIZE, 2]
    g_icov = inv_cov2D.reshape(-1, 4)[tile_gids] # [num_tiles, BLOCK_SIZE, 4]
    g_ops = sig_opacities[tile_gids]        # [num_tiles, BLOCK_SIZE, 1]
    g_cols = colors[tile_gids]              # [num_tiles, BLOCK_SIZE, 3]
    
    # Pre-calculate global pixel coordinates for EVERY pixel in the image, 
    # grouped by tile. Shape: [num_tiles, 256] (since 16x16 = 256)
    idx = jnp.arange(256, dtype=jnp.int32)
    tx = jnp.arange(num_tiles) % num_tiles_x
    ty = jnp.arange(num_tiles) // num_tiles_x
    
    px = (tx[:, None] * TILE_SIZE).astype(jnp.float32) + (idx % TILE_SIZE)[None, :].astype(jnp.float32) + 0.5
    py = (ty[:, None] * TILE_SIZE).astype(jnp.float32) + (idx // TILE_SIZE)[None, :].astype(jnp.float32) + 0.5
    
    @jax.checkpoint
    def scan_fn(carry, i):
        """
        The core alpha-blending loop.
        Processes the i-th Gaussian for ALL pixels in ALL tiles simultaneously.
        """
        c_accum, T = carry
        
        # Extract the parameters for the i-th Gaussian across all tiles
        mu_x = g_means[:, i, 0][:, None] # [num_tiles, 1]
        mu_y = g_means[:, i, 1][:, None]
        ic00 = g_icov[:, i, 0][:, None]
        ic01 = g_icov[:, i, 1][:, None]
        ic11 = g_icov[:, i, 3][:, None]
        op = g_ops[:, i, 0][:, None]
        
        # Vectorized quadratic form: (x-mu)^T * Sigma^-1 * (x-mu)
        dx = px - mu_x # [num_tiles, 256]
        dy = py - mu_y
        
        power = -0.5 * (dx * dx * ic00 + 2.0 * dx * dy * ic01 + dy * dy * ic11)
        
        # Prevent NaNs in exp() by masking out power for inactive padded elements
        is_active_local = local_mask[:, i][:, None]
        power = jnp.where(is_active_local, power, -100.0)
        
        # Compute alpha influence
        alpha = jnp.where(power > -10.0, jnp.exp(jnp.clip(power, -100.0, 0.0)) * op, 0.0)
        
        # Combine local existence mask and transmittance threshold
        is_active = is_active_local & (T > 1e-4)
        alpha = jnp.where(is_active, jnp.minimum(0.99, alpha), 0.0)
        
        # Update color and transmittance
        weight = alpha * T
        c_accum = c_accum + weight[:, :, None] * g_cols[:, i, :][:, None, :]
        T = T * (1.0 - alpha)
        
        return (c_accum, T), None
        
    # Initialize accumulation buffers
    c_init = jnp.zeros((num_tiles, 256, 3), dtype=jnp.float32)
    T_init = jnp.ones((num_tiles, 256), dtype=jnp.float32)
    
    # Run the optimized scan loop
    (final_c, final_T), _ = jax.lax.scan(scan_fn, (c_init, T_init), jnp.arange(BLOCK_SIZE))
    
    # Add the background color based on remaining transmittance
    final_color = final_c + final_T[:, :, None] * background[None, None, :]
    
    # Reshape the flat tile-pixel dimension [num_tiles, 256] back into an image [H, W, 3]
    out_grid = final_color.reshape(num_tiles_y, num_tiles_x, TILE_SIZE, TILE_SIZE, 3)
    out_image = out_grid.swapaxes(1, 2).reshape(num_tiles_y * TILE_SIZE, num_tiles_x * TILE_SIZE, 3)
    
    return out_image[:H, :W, :]
