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
    Simplified version for Gaussian Splatting (fixed window, uniform kernel).
    """
    channel = img1.shape[-1]
    window = jnp.ones((window_size, window_size, 1, channel)) / (window_size * window_size)
    
    mu1 = jax.lax.conv_general_dilated(
        img1[None], window, (1, 1), 'SAME', 
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=channel
    )[0]
    mu2 = jax.lax.conv_general_dilated(
        img2[None], window, (1, 1), 'SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=channel
    )[0]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = jnp.maximum(0, jax.lax.conv_general_dilated(
        (img1 * img1)[None], window, (1, 1), 'SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=channel
    )[0] - mu1_sq)
    sigma2_sq = jnp.maximum(0, jax.lax.conv_general_dilated(
        (img2 * img2)[None], window, (1, 1), 'SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=channel
    )[0] - mu2_sq)
    sigma12 = jax.lax.conv_general_dilated(
        (img1 * img2)[None], window, (1, 1), 'SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=channel
    )[0] - mu1_mu2

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
