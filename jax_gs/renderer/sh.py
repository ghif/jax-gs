import jax.numpy as jnp
import jax

# Spherical harmonics coefficients for degrees 0 to 3
# Source: Standard 3DGS implementation
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

def eval_sh(sh_degree: int, sh_coeffs: jnp.ndarray, view_dirs: jnp.ndarray):
    """
    Evaluate spherical harmonics to get view-dependent colors.
    
    Args:
        sh_degree: Current SH degree being used (0-3).
        sh_coeffs: SH coefficients of shape (N, K, 3).
        view_dirs: Normalized viewing directions from the Gaussian means to the camera center, shape (N, 3).
    Returns:
        colors: View-dependent colors, shape (N, 3).
    """
    # sh_coeffs shape is usually (N, (max_degree+1)**2, 3)
    # We sum up the contributions from degree 0 to sh_degree
    
    # Degree 0 (DC)
    result = C0 * sh_coeffs[:, 0]
    
    if sh_degree > 0:
        x, y, z = view_dirs[:, 0:1], view_dirs[:, 1:2], view_dirs[:, 2:3]
        
        # Degree 1
        # Indices: 1 (y), 2 (z), 3 (x)
        result = result - C1 * y * sh_coeffs[:, 1] + C1 * z * sh_coeffs[:, 2] - C1 * x * sh_coeffs[:, 3]

        if sh_degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            
            # Degree 2
            # Indices: 4 (xy), 5 (yz), 6 (2zz - xx - yy), 7 (xz), 8 (xx - yy)
            result = result + \
                C2[0] * xy * sh_coeffs[:, 4] + \
                C2[1] * yz * sh_coeffs[:, 5] + \
                C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6] + \
                C2[3] * xz * sh_coeffs[:, 7] + \
                C2[4] * (xx - yy) * sh_coeffs[:, 8]

            if sh_degree > 2:
                # Degree 3
                # Indices: 9, 10, 11, 12, 13, 14, 15
                result = result + \
                    C3[0] * y * (3.0 * xx - yy) * sh_coeffs[:, 9] + \
                    C3[1] * xy * z * sh_coeffs[:, 10] + \
                    C3[2] * y * (4.0 * zz - xx - yy) * sh_coeffs[:, 11] + \
                    C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[:, 12] + \
                    C3[4] * x * (4.0 * zz - xx - yy) * sh_coeffs[:, 13] + \
                    C3[5] * z * (xx - yy) * sh_coeffs[:, 14] + \
                    C3[6] * x * (xx - 3.0 * yy) * sh_coeffs[:, 15]
                
    return result
