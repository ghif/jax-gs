# Building a High-Performance Tile-Based Rasterizer with MLX

In this post, we’ll dive into the implementation of a high-performance tile-based rasterizer for Gaussian Splatting, specifically tailored for Apple Silicon using the **MLX** framework. We'll break down the two main stages: **Interaction Generation** and **Parallel Rendering**.

## The Problem: Rendering Millions of Splats
Gaussian Splatting involves projecting 3D Gaussians onto a 2D image plane. Each splat contributes to the final color of a pixel based on its position, covariance (shape), and opacity. For an image with millions of splats, a naive approach (checking every splat for every pixel) is $O(N \cdot P)$, which is prohibitively slow.

Instead, we use **Tile-Based Rasterization**. We divide the image into $16 \times 16$ tiles and only process the splats that actually overlap a specific tile.

---

## Why MLX? The Power of Apple Silicon
MLX is a machine learning framework designed by Apple researchers specifically for Apple Silicon. It stands out from other frameworks on macOS due to several key architectural advantages:

*   **Unified Memory Architecture (UMA):** In M-series chips, the CPU and GPU share the same physical memory pool. MLX leverages this with a "zero-copy" model—data can be accessed by both processors without expensive transfers or synchronization overhead.
*   **Direct Metal Integration:** MLX kernels are built directly on top of **Metal**, Apple's low-level API. This ensures that every operation is hand-tuned for the execution units of Apple's GPU.
*   **Lazy Evaluation:** MLX uses a lazy evaluation model that builds a dynamic graph and only executes it when a value is needed (e.g., via `mx.eval()`). This allows for runtime graph optimizations that are particularly effective for the complex control flows in Gaussian Splatting.

---

## Stage 1: Gaussian Projection
Before we can rasterize, we must project our 3D Gaussians into 2D screen space. This involves several coordinate transformations and the calculation of 2D covariance matrices.

### 1. World to Camera Space
We transform the 3D means from world coordinates to camera coordinates using the World-to-Camera ($W2C$) matrix:
$$ \mu_{cam} = (W2C) \mu_{world} $$

### 2. Differentiable Projection (The Jacobian)
To project the 3D covariance $\Sigma$ into 2D space $\Sigma'$, we use a linear approximation of the projection function via its Jacobian $J$:
$$ \Sigma' = J W \Sigma W^T J^T $$
Where $W$ is the rotation component of the viewing transformation. The Jacobian $J$ for a pinhole camera is:
$$ J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix} $$

In MLX, we implement this as a differentiable operation:
```python
# Jacobian construction
J = mx.stack([
    mx.stack([fx / z, mx.zeros_like(z), -fx * x / (z**2)], axis=-1),
    mx.stack([mx.zeros_like(z), fy / z, -fy * y / (z**2)], axis=-1)
], axis=-2)
```

### 3. Low-Pass Filtering
To prevent aliasing, we add a small 0.3 pixel bias to the diagonal of the 2D covariance, effectively acting as a low-pass filter:
$$ \Sigma'_{filtered} = \Sigma' + 0.3 I $$

---

## Stage 2: Tile Interaction Generation
The goal of this stage is to create a sorted list of which Gaussians belong to which tiles.

### 1. Binning
First, we calculate the bounding box of each Gaussian in screen space and determine which tiles it intersects. We use the Gaussian's radius (derived from its 2D covariance eigenvalues) to define its reach.

```python
# Calculate screen-space bounding box and tile ranges
min_x = mx.clip((means2D[:, 0] - radii), 0, W - 1)
max_x = mx.clip((means2D[:, 0] + radii), 0, W - 1)
min_y = mx.clip((means2D[:, 1] - radii), 0, H - 1)
max_y = mx.clip((means2D[:, 1] + radii), 0, H - 1)

tile_min_x = (min_x // tile_size).astype(mx.int32)
tile_max_x = (max_x // tile_size).astype(mx.int32)
tile_min_y = (min_y // tile_size).astype(mx.int32)
tile_max_y = (max_y // tile_size).astype(mx.int32)
```

### 2. Sorting (The Depth Secret)
To render correctly, we must process Gaussians from front to back. We combine the **Tile ID** and the **Depth** into a single 64-bit key:

$$ \text{Key} = (\text{Tile ID} \ll 32) \lor \text{Quantized Depth} $$

By sorting these keys using `mx.argsort`, we get a list that is primary-sorted by tile and secondary-sorted by depth.

```python
# Construct 31-bit Key for sorting
key = (sort_tile_ids.astype(mx.uint32) << 13) | depth_quant.astype(mx.uint32)
sort_indices = mx.argsort(key)
```

---

## Stage 3: Parallel Tile Rendering
Once we have our sorted interactions, we render each tile independently. In MLX, we use `mx.vmap` to parallelize this across the GPU.

### 1. The Gaussian PDF
For a pixel at position $x$, the contribution of a Gaussian with mean $\mu$ and covariance $\Sigma$ is determined by:

$$ f(x) = \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right) $$

In the code, we compute the "power" term $(x - \mu)^T \Sigma^{-1} (x - \mu)$ differentiably:

```python
dx = grid_x - t_means[:, 0]
dy = grid_y - t_means[:, 1]
# Quadratic form: dx^2 * inv_cov[0,0] + 2 * dx * dy * inv_cov[0,1] + dy^2 * inv_cov[1,1]
pow_val = -0.5 * (dx * row0 + dy * row1)
alpha = mx.exp(pow_val) * opacities
```

### 2. Alpha Blending and Transmittance
We use the standard volume rendering equation. The final color $C$ of a pixel is a weighted sum of the colors $c_i$ of the Gaussians overlapping it:

$$ C = \sum_{i=1}^N c_i \alpha_i T_i $$

where $T_i$ is the **transmittance**, representing how much light reaches the $i$-th Gaussian after passing through those in front of it:

$$ T_i = \prod_{j=1}^{i-1} (1 - \alpha_j) $$

In MLX, we compute the effective alpha ($\alpha_{eff}$) by combining the Gaussian's spatial influence with its learned opacity:

```python
# Calculate effective alpha for each pixel in the tile
alpha = mx.exp(pow_val) * mx.expand_dims(mx.expand_dims(t_ops, -1), -1)
mask = mx.expand_dims(mx.expand_dims(local_mask, -1), -1) & (pow_val > -10.0)
alpha_eff = mx.where(mask, mx.minimum(0.99, alpha), 0.0)
```

### 3. MLX Optimization: Vectorized Blending
Unlike traditional implementations that use a sequential loop (like `jax.lax.scan`), our MLX rasterizer is fully vectorized within a block. We use `mx.cumprod` to calculate transmittance for all Gaussians in a single pass:

```python
# Vectorized Transmittance and Color Accumulation
one_minus_alpha = 1.0 - alpha_eff
T = mx.concatenate([
    mx.ones((1, tile_size, tile_size)), 
    mx.cumprod(one_minus_alpha[:-1], axis=0)
], axis=0)

weights = alpha_eff * T
accum_color = mx.sum(mx.expand_dims(weights, -1) * t_cols[:, None, None, :], axis=0)

# Final background blending
final_T = T[-1] * one_minus_alpha[-1]
final_tile = accum_color + mx.expand_dims(final_T, -1) * background
```

By processing a `BLOCK_SIZE` (e.g., 192) of Gaussians at once, we saturate the GPU's SIMD units much more effectively than a sequential loop would.

---

## The Complete Rendering Pipeline
The `render_mlx` function orchestrates these stages into a seamless pipeline. It handles Spherical Harmonics (SH) to RGB conversion, triggers the projection, generates tile interactions, and finally calls the vectorized rasterizer.

```python
def render_mlx(params: dict, camera_dict: dict):
    # 1. Project 3D Gaussians to 2D
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(params, camera_dict)
    
    # 2. SH-to-RGB Conversion
    colors = sh_to_rgb(params["sh_coeffs"])
    
    # 3. Generate Tile Interactions (Sort by TileID and Depth)
    sorted_tile_ids, sorted_gaussian_ids, _ = get_tile_interactions(...)
    
    # 4. Render Tiles (Parallelized via vmap)
    image = render_tiles(...)
    
    return image
```

---

## Conclusion
By combining efficient bit-packed sorting with vectorized tile rendering, the MLX rasterizer achieves incredible throughput on Apple Silicon. The use of `vmap` allows the framework to handle the complex scheduling of tile-based work, while `cumprod` unlocks the massive parallel potential of the M-series GPU.

```