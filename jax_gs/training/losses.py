import jax
import jax.numpy as jnp

def l1_loss(pred, target):
    """
    Mean Absolute Error.
    """
    return jnp.mean(jnp.abs(pred - target))

def mse_loss(pred, target):
    """
    Mean Squared Error.
    """
    return jnp.mean((pred - target) ** 2)

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Structural Similarity Index Measure.
    Separable version for Gaussian Splatting (fixed window, uniform kernel).
    """
    channel = img1.shape[-1]
    
    # Separable Gaussian Window (1D)
    window_1d = jnp.ones((window_size, 1, 1, channel)) / window_size
    
    def blur(img):
        # Vertical pass
        h = jax.lax.conv_general_dilated(
            img[None], window_1d, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=channel
        )
        # Horizontal pass (using transposed kernel)
        window_1d_h = window_1d.transpose(1, 0, 2, 3)
        return jax.lax.conv_general_dilated(
            h, window_1d_h, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=channel
        )[0]
    
    mu1 = blur(img1)
    mu2 = blur(img2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = jnp.maximum(0, blur(img1 * img1) - mu1_sq)
    sigma2_sq = jnp.maximum(0, blur(img2 * img2) - mu2_sq)
    sigma12 = blur(img1 * img2) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = jnp.clip(ssim_map, -1.0, 1.0)

    if size_average:
        return jnp.mean(ssim_map)
    else:
        return jnp.mean(ssim_map, axis=(0, 1, 2))

def d_ssim_loss(pred, target):
    """
    Structural Dissimilarity loss.
    """
    return jnp.maximum(0, (1.0 - ssim(pred, target)) / 2.0)

def depth_distortion_loss(depth, depth_sq, accum_weight):
    """
    Computes a simplified depth distortion loss using depth variance.
    This encourages splats along a ray to concentrate at a single depth.

    Args:
        depth: (H, W, 1) Rendered depth map
        depth_sq: (H, W, 1) Rendered depth squared map
        accum_weight: (H, W, 1) Accumulated weight map
    Returns:
        loss: Scalar loss value
    """
    # Use stop_gradient on the weight to avoid pushing opacities to zero
    w = jax.lax.stop_gradient(accum_weight)
    
    # Safe normalization: only normalize where we have significant weight
    # to avoid numerical instability in the gradient.
    mask = w > 1e-4
    safe_w = jnp.where(mask, w, 1.0)
    
    # Normalize depth expectations
    exp_d = jnp.where(mask, depth / safe_w, 0.0)
    exp_d_sq = jnp.where(mask, depth_sq / safe_w, 0.0)
    
    # Variance = E[d^2] - (E[d])^2
    # Ensure it's non-negative for numerical safety
    variance = jnp.maximum(exp_d_sq - exp_d**2, 0.0)
    
    # Weight by accum_weight so empty areas have zero loss
    return jnp.mean(variance * w)

def normal_consistency_loss(rendered_normals, depth_map, camera):
    """
    Computes normal consistency loss between rendered normals and 
    normals derived from the depth map gradient.

    Args:
        rendered_normals: (H, W, 3) Accumulated normal map
        depth_map: (H, W, 1) Accumulated depth map
        camera: Camera object for intrinsics
    Returns:
        loss: Scalar loss value
    """
    # 1. Compute surface normal from depth map gradient
    # Use stop_gradient on depth_map for the gradient calculation to avoid 
    # pushing opacities to zero via the depth gradient.
    d = jax.lax.stop_gradient(depth_map)
    
    # Simple central difference for gradients
    dz_dx = (jnp.roll(d, -1, axis=1) - jnp.roll(d, 1, axis=1)) / 2.0
    dz_dy = (jnp.roll(d, -1, axis=0) - jnp.roll(d, 1, axis=0)) / 2.0

    # Simplified version in camera space:
    nx = -dz_dx
    ny = -dz_dy
    nz = jnp.ones_like(d)

    n_depth = jnp.concatenate([nx, ny, nz], axis=-1)
    # Safe normalization
    n_depth_norm = jnp.sqrt(jnp.sum(n_depth**2, axis=-1, keepdims=True) + 1e-6)
    n_depth = n_depth / n_depth_norm

    # 2. Compute consistency with rendered normals
    rendered_norm = jnp.sqrt(jnp.sum(rendered_normals**2, axis=-1, keepdims=True) + 1e-8)
    
    # Safe unit normal for rendered splats
    n_rendered = rendered_normals / rendered_norm
    
    # Cosine similarity
    cos_sim = jnp.sum(n_rendered * n_depth, axis=-1, keepdims=True)
    
    # Weighted loss: (1 - cos_sim) * stop_gradient(rendered_norm)
    loss_map = (1.0 - cos_sim) * jax.lax.stop_gradient(rendered_norm)
    
    return jnp.mean(loss_map)