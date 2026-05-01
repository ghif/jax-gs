# Understanding the JAX 3DGS Rasterizer

This document explains the tiled rasterization algorithm implemented in `jax_gs/renderer/rasterizer.py`. This is the core "engine" that turns a collection of 3D Gaussians into a 2D image.

The algorithm works in two major stages: **Tile Interaction Generation** and **Tile Rendering**.

---

## Stage 1: Tile Interaction Generation (`get_tile_interactions`)

In 3D Gaussian Splatting, we don't check every Gaussian for every pixel (which would be incredibly slow). Instead, we divide the image into a grid of **16x16 tiles**.

### 1. Bounding Boxes
For every Gaussian, we calculate which tiles it covers. We look at the 2D center and its "radius" (how far it spreads) to find the `min_x`, `max_x`, `min_y`, and `max_y` in tile coordinates.

### 2. Gaussian-to-Tile Mapping
If a Gaussian is large, it might overlap multiple tiles. We create a list of "interactions" where each entry is a pair: `(Tile ID, Gaussian ID)`.
*   If Gaussian #5 covers tiles #1, #2, and #10, we generate three interaction entries.
*   To keep this efficient in JAX, we use an 8x8 meshgrid to check up to 64 possible tiles per Gaussian in parallel.

### 3. The "Pack-Sort" Trick
To render correctly, Gaussians must be processed from **front to back** (nearest to the camera first). We need to sort our interactions by `Tile ID` (so all Gaussians for one tile are together) and then by `Depth`.

Instead of sorting twice, we use **Bit-packing**:
1.  We take the `Tile ID` and shift its bits to the left.
2.  We put the `Depth` (converted to a small integer) in the remaining right-side bits.
3.  We sort this single integer. Because the `Tile ID` is in the higher bits, the list is grouped by tile first, and then perfectly ordered by depth within each tile.

---

## Stage 2: Tile Rendering (`render_tiles`)

Once we have a sorted list of which Gaussians belong to which tile, we can render all tiles at the same time using `jax.vmap`.

### 1. Tile Boundaries
We use `jnp.searchsorted` to find exactly where in our big sorted list each tile starts and ends. This tells the renderer: "Tile #5, your Gaussians start at index 100 and end at index 150."

### 2. The `scan` Loop (Alpha Blending)
For each pixel inside a tile, we iterate through the Gaussians assigned to that tile. In JAX, we use `jax.lax.scan` for this loop because it is very efficient for "carrying state" (like the current color and transparency) through a sequence.

For every Gaussian, we calculate its contribution to the pixel:
1.  **Distance Check:** How far is this pixel from the center of the Gaussian? We use the "Quadratic Form" (the covariance matrix) to find the intensity.
2.  **Alpha calculation:** We multiply the intensity by the Gaussian's opacity.
3.  **Blending (Over-compositing):**
    *   `new_color = accumulated_color + (alpha * current_transmittance) * color`
    *   `new_transmittance = current_transmittance * (1.0 - alpha)`

### 3. Early Termination (Logic)
We keep track of `T` (Transmittance). `T` starts at 1.0 (fully transparent) and goes toward 0.0 (fully opaque). If `T` becomes very small (e.g., `< 0.0001`), it means the pixel is already "full," and Gaussians behind it won't be visible. We stop adding color to save computation.

---

## Why this is fast in JAX

1.  **Parallelism:** `jax.vmap` allows us to compute every tile in the image at the exact same time on the GPU/TPU.
2.  **JIT Compilation:** The entire logic is compiled into a single optimized kernel.
3.  **No "Gaps":** By using `BLOCK_SIZE` (fixed-size chunks), we ensure the GPU always has a predictable amount of work, which prevents the hardware from idling.

---

## Key Terms for Beginners
*   **Splat:** The 2D projection of a 3D Gaussian on the screen.
*   **Tile:** A small square section of the image (usually 16x16 pixels).
*   **Transmittance (T):** How much light can still pass through the "fog" of Gaussians.
*   **Alpha:** The opacity of a single Gaussian at a specific pixel.

---

## The Path Forward: Accelerating Rasterization with JAX Pallas

While the current implementation using `jax.vmap` and `jax.lax.scan` is heavily optimized for general-purpose compilation (XLA), it still suffers from some hardware-level bottlenecks on accelerators like TPUs. The sequence of memory reads and writes (loading Gaussian data, updating running pixel colors) can overwhelm the memory bandwidth. 

To break past these limitations, we can rewrite the tile rendering logic using **JAX Pallas**, a framework for writing custom, hardware-level kernels (such as Mosaic for TPUs).

### 1. Custom Memory Management (VMEM)
Instead of relying on the XLA compiler to guess the best memory layout, Pallas allows us to explicitly manage the TPU's ultra-fast Vector Memory (VMEM).
*   **The Strategy:** We allocate a `[16, 16, 3]` color buffer and a `[16, 16, 1]` transmittance buffer directly in VMEM for each tile. 
*   **The Benefit:** As we blend Gaussians into the tile, all math happens instantly in VMEM. We only write the final `16x16` tile back to the much slower High Bandwidth Memory (HBM) *once* at the very end.

### 2. Blocked Gaussian Streaming
Currently, we gather all Gaussians for a tile at once (up to `BLOCK_SIZE`), which requires massive memory reads.
*   **The Strategy:** Using Pallas, we can use `pl.load` to stream in small "chunks" (e.g., 16 or 32) of Gaussians from HBM into VMEM. 
*   **The Benefit:** This keeps the memory pipeline fed constantly without overwhelming the registers, maximizing the compute utilization of the Vector Processing Units (VPUs).

### 3. Tile-Level Early Ray Termination
In the current `jax.lax.scan` loop, every pixel in a tile processes all `BLOCK_SIZE` Gaussians, even if the pixel is already fully opaque (`T < 1e-4`). JAX masks the math, but the GPU/TPU still does the work.
*   **The Strategy:** Inside the Pallas kernel, after processing a chunk of Gaussians, we perform a fast reduction (e.g., `pl.max`) to find the highest transmittance value left across the entire `16x16` tile. 
*   **The Benefit:** If the maximum `T` drops below the threshold, it means *every* pixel in the tile is fully opaque. We can instantly `break` out of the loop and move to the next tile, saving massive amounts of compute on dense foreground scenes.

### 4. Sparse Interactions Buffer
The `get_tile_interactions` function currently creates a massive intermediate array (e.g., 3.2 million elements for 50k points) to find which Gaussians intersect which tiles before sorting them.
*   **The Strategy:** We can write a custom Pallas kernel that calculates bounding boxes and streams valid `(Tile ID, Gaussian ID)` pairs directly into a pre-allocated sparse memory buffer in a two-pass approach.
*   **The Benefit:** This completely eliminates the dense $N \times 64$ memory overhead, allowing the pipeline to scale to millions of Gaussians without running out of memory.

**Summary:** By migrating the core loops to Pallas, we transition from relying on XLA's generic fusion to a customized, hardware-aware pipeline. This unlocks SRAM/VMEM utilization and early termination, bridging the gap between high-level JAX and handwritten CUDA/Mosaic kernels.
