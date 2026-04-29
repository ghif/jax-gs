# Beyond the Cloud: The Shift to 2D Gaussian Splatting

## Abstract

Since its inception, 3D Gaussian Splatting (3DGS) has revolutionized the field of neural rendering, offering real-time, high-fidelity scene reconstruction from sparse images. However, 3DGS often struggles with thin surfaces and view-dependent artifacts due to its volumetric nature. In this article, we explore the evolution of this technique into **2D Gaussian Splatting (2DGS)**. By constraining primitives to oriented 2D disks, 2DGS provides a more geometrically consistent representation of surfaces, enabling sharper reconstructions and explicit normal estimation.

## Introduction

Gaussian Splatting represents a scene as a collection of millions of flexible, semi-transparent kernels. In the original 3D formulation, these kernels are ellipsoids characterized by a 3D covariance matrix. While powerful, these "3D clouds" often fail to accurately model thin surfaces—like walls or cloth—resulting in "pancaking" artifacts or volumetric "fuzziness" when viewed from steep angles.

2D Gaussian Splatting addresses this by redefining the primitives as **oriented 2D disks**. This shift from volumes to surfaces not only reduces the parameter count (two scales instead of three) but also enforces a hard geometric constraint that aligns better with the physical world's surfaces.

## The Geometry of 2D Gaussians

In 2DGS, each primitive is a flat disk in 3D space. Mathematically, a 2D Gaussian is defined by its center $\mu \in \mathbb{R}^3$ and two tangent vectors $u, v \in \mathbb{R}^3$ that span the disk's plane.

### The Transformation Matrix
Unlike 3DGS, which uses a $3 \times 3$ covariance matrix, 2DGS uses a $3 \times 2$ mapping matrix $M$ that transforms a local 2D coordinate $(x, y)$ on the disk to a 3D point:

$$M = [s_1 u \mid s_2 v]$$

where $s_1, s_2$ are the learned scales along the tangent directions. The vectors $u, v$ are derived from a quaternion $q$, ensuring they are orthonormal. The normal vector $n$ is simply the cross product $u \times v$.

## Perspective-Correct Projection

To render these disks onto a 2D screen, we must project them using the camera's intrinsic and extrinsic parameters.

### The Jacobian of Projection
Let $W$ be the world-to-camera transformation. The position of a Gaussian in camera space is $x_{cam} = W \mu$. The perspective projection $\pi(x)$ maps this to screen coordinates. To handle the "spread" of the disk, we use the Jacobian $J$ of the projection function at $x_{cam}$:

$$J = \frac{\partial \pi}{\partial x} = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix}$$

### 2D Covariance
The projected 2D covariance $C_{2D}$ is computed by projecting the 3D tangent vectors into the 2D image plane:

$$M_{2D} = J \cdot R \cdot M$$

where $R$ is the camera rotation. The resulting $2 \times 2$ covariance matrix is:

$$C_{2D} = M_{2D} M_{2D}^T + \nu I$$

The term $\nu I$ acts as a low-pass filter (typically $\nu=0.3$) to prevent aliasing when the splat becomes smaller than a pixel.

## Rendering: Alpha Blending

The rendering process follows the classic "Tile-based Rasterization" approach. For each pixel, we find all overlapping Gaussians, sort them by depth, and perform front-to-back alpha blending:

$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

where $\alpha_i$ is the opacity of the $i$-th Gaussian scaled by its spatial influence:

$$\alpha_i = \sigma_i \exp\left(-\frac{1}{2} (x - \mu_{2D})^T C_{2D}^{-1} (x - \mu_{2D})\right)$$

## Why 2DGS? The Benefits

### 1. Superior Surface Reconstruction
By forcing Gaussians to be flat, 2DGS naturally avoids the "volumetric thickness" issue of 3DGS. Surfaces appear as sharp interfaces rather than semi-transparent fog.

### 2. Explicit Normals
Since each 2D Gaussian has a well-defined normal $n = u \times v$, we can render a **normal map** alongside the color image. This allows for sophisticated regularization, such as the **Normal Consistency Loss**:

$$\mathcal{L}_{normal} = 1 - \cos(\mathbf{n}_{rendered}, \mathbf{n}_{depth})$$

where $\mathbf{n}_{depth}$ is the normal derived from the gradient of the rendered depth map. This ensures the geometric normals of the splats align with the overall surface topology.

### 3. View-Angle Consistency
3D ellipsoids often look different when sliced by rays at different angles. 2D disks, being infinitesimal surfaces, provide more consistent color and opacity transitions, especially for specular materials.

## Conclusion

2D Gaussian Splatting represents a significant step towards bridging the gap between point-based rendering and traditional mesh-based graphics. By embracing the 2D nature of surfaces, it provides a more robust and geometrically meaningful representation of the 3D world. Whether you are building a digital twin or a high-end visual effect, 2DGS offers the sharpness and consistency that 3D clouds often lack.

---
*This article was generated based on the `jax-gs` implementation of 2D Gaussian Splatting.*
