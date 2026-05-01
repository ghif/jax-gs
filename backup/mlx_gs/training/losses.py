import mlx.core as mx
import numpy as np

def l1_loss(pred, target):
    """
    Mean Absolute Error in MLX.
    """
    return mx.mean(mx.abs(pred - target))

def mse_loss(pred, target):
    """
    Mean Squared Error in MLX.
    """
    return mx.mean(mx.square(pred - target))

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Structural Similarity Index Measure in MLX using separable 1D convolutions.
    """
    # img1, img2: (H, W, C)
    H, W, C = img1.shape
    
    # win1d: (window_size,)
    win1d = mx.ones((window_size,)) / window_size
    # win_h: (1, 1, window_size, 1)
    win_h = win1d[None, None, :, None]
    # win_v: (1, window_size, 1, 1)
    win_v = win1d[None, :, None, None]
    
    # Repeat for channels
    win_h = mx.tile(win_h, (C, 1, 1, 1))
    win_v = mx.tile(win_v, (C, 1, 1, 1))
    
    # Add batch dimension: (1, H, W, C)
    img1_b = mx.expand_dims(img1, 0)
    img2_b = mx.expand_dims(img2, 0)
    
    def conv_separable(x):
        # Horizontal pass
        x = mx.conv2d(x, win_h, groups=C, padding=(0, window_size//2))
        # Vertical pass
        x = mx.conv2d(x, win_v, groups=C, padding=(window_size//2, 0))
        return x

    mu1 = conv_separable(img1_b)[0]
    mu2 = conv_separable(img2_b)[0]

    mu1_sq = mx.square(mu1)
    mu2_sq = mx.square(mu2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = mx.maximum(0, conv_separable(mx.expand_dims(mx.square(img1), 0))[0] - mu1_sq)
    sigma2_sq = mx.maximum(0, conv_separable(mx.expand_dims(mx.square(img2), 0))[0] - mu2_sq)
    sigma12 = conv_separable(mx.expand_dims(img1 * img2, 0))[0] - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = mx.clip(ssim_map, -1.0, 1.0)

    if size_average:
        return mx.mean(ssim_map)
    else:
        return mx.mean(ssim_map, axis=(0, 1, 2))

def d_ssim_loss(pred, target):
    """
    Structural Dissimilarity loss in MLX.
    """
    return mx.maximum(0.0, (1.0 - ssim(pred, target)) / 2.0)
