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

def depth_distortion_loss(depth, depth_sq):
    """
    Computes a simplified depth distortion loss using depth variance.
    This encourages splats along a ray to concentrate at a single depth.

    Args:
        depth: (H, W, 1) Rendered depth map
        depth_sq: (H, W, 1) Rendered depth squared map
    Returns:
        loss: Scalar loss value
    """
    # Variance = E[d^2] - (E[d])^2
    # For a single ray, this is proportional to the distortion loss
    variance = jnp.maximum(depth_sq - depth**2, 0.0)
    return jnp.mean(variance)

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
    # Normalize depth map for gradient calculation
    H, W = depth_map.shape[:2]

    # Simple central difference for gradients
    dz_dx = (jnp.roll(depth_map, -1, axis=1) - jnp.roll(depth_map, 1, axis=1)) / 2.0
    dz_dy = (jnp.roll(depth_map, -1, axis=0) - jnp.roll(depth_map, 1, axis=0)) / 2.0

    # Backproject pixels to 3D to get surface normals
    # For a pinhole camera: N = normalize([-dz/dx * f/x, -dz/dy * f/y, 1])
    # Simplified version in camera space:
    nx = -dz_dx
    ny = -dz_dy
    nz = jnp.ones_like(depth_map)

    n_depth = jnp.concatenate([nx, ny, nz], axis=-1)
    n_depth = n_depth / jnp.linalg.norm(n_depth, axis=-1, keepdims=True)

    # 2. Compute consistency with rendered normals
    # Use a weighted version of the loss to avoid division by zero and exploding gradients.
    # The weight is the norm of the accumulated normals (related to opacity).
    rendered_norm = jnp.linalg.norm(rendered_normals, axis=-1, keepdims=True)
    
    # Safe unit normal for rendered splats - use 'double where' trick for JAX stability
    safe_norm = jnp.where(rendered_norm > 1e-6, rendered_norm, 1.0)
    n_rendered = jnp.where(rendered_norm > 1e-6, rendered_normals / safe_norm, 0.0)
    
    # Cosine similarity
    cos_sim = jnp.sum(n_rendered * n_depth, axis=-1, keepdims=True)
    
    # Weighted loss: (1 - cos_sim) * stop_gradient(rendered_norm)
    # We stop gradient on the weight so the loss only optimizes normal directions,
    # not the existence/opacity of the splats themselves.
    loss_map = (1.0 - cos_sim) * jax.lax.stop_gradient(rendered_norm)
    
    return jnp.mean(loss_map)