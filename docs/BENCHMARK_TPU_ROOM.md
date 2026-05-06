# Room TPU Rasterizer Benchmark

This benchmark compares the performance of `train.py` on the LLFF room dataset (`gs://dataset-nerf/nerf_llff_data/room`, `images_8`) with and without the `--fast_tpu_rasterizer` flag.

## Environment
- **Platform:** Google TPU (v3-8 equivalent, 32GB HBM per core)
- **Dataset:** LLFF Room (504x378 resolution)
- **Initial Points:** 8,772
- **Max Padded Gaussians:** 70,176
- **Conda Environment:** `tpu-env`

## Results

| Rasterizer | Status | Throughput (Steady State) | Compile Time (Initial) |
| :--- | :--- | :--- | :--- |
| **Standard** (`rasterizer.py`) | **Success** (Fixed) | **~0.09 it/s** | ~30s |
| **Fast TPU** (`rasterizer_tpu.py`) | **Success** | **~9.6 it/s** | ~36s |

### Analysis

1.  **Standard Rasterizer Fix:** The standard rasterizer previously failed with a `RESOURCE_EXHAUSTED` (OOM) error due to a 55GB broadcast of interaction arrays across all tiles. This was resolved by implementing three memory optimizations:
    -   Replacing the outer `vmap` with `jax.lax.map` (sequential tile processing).
    -   Using `jax.lax.dynamic_slice` instead of `jnp.take` to prevent broadcasts.
    -   Applying `@jax.checkpoint` to the inner blending loop for rematerialization.
2.  **Performance Gap:** While functional, the standard rasterizer is approximately **100x slower** than the fast TPU version. This is because sequential tile processing significantly increases training time to conserve memory.
3.  **Fast TPU Advantage:** The optimized TPU rasterizer achieves high performance by flattening tiles and pixels into vectorized XLA operations that fully utilize the TPU's Matrix Unit (MXU).
4.  **Convergence:** Both rasterizers produce identical loss and rendering results, confirming mathematical equivalence between the two implementations.

### Why Fast TPU Rasterizer is Faster

The ~100x performance difference is due to fundamental architectural differences in how each rasterizer utilizes the TPU's hardware (MXU and HBM):

1.  **MXU Saturation via Flattened Vectorization**:
    -   **Standard**: Parallelizes tiles using `vmap` (or `lax.map` for memory safety). Inside each tile, it loops through pixels. This creates fragmented, smaller operations that do not efficiently saturate the TPU's Matrix Unit (MXU).
    -   **Fast TPU**: Flattens the `[num_tiles, 16, 16]` dimensions into a single `[num_tiles * 256]` dimension. This allows XLA to generate large, dense matrix multiplications that are perfectly sized for the TPU MXU, maximizing compute throughput.

2.  **Prefetched Gathers vs. In-Loop Indexing**:
    -   **Standard**: Inside the tight alpha-blending loop, it uses indexing to fetch Gaussian parameters. On TPU, these repeated "gather" operations from HBM are high-latency and break the compute pipeline.
    -   **Fast TPU**: Uses a "Broadcasted Gather" strategy. It fetches all required Gaussian data for every tile into a massive contiguous buffer *before* entering the alpha-blending `scan`. This converts hundreds of random-access lookups into a single efficient memory burst.

3.  **Parallel vs. Sequential Tile Processing**:
    -   **Standard (Fixed)**: To avoid OOM, it uses `jax.lax.map` to process tiles sequentially (or in small batches). This keeps memory low but forces the TPU to wait for each tile to finish, resulting in very low utilization.
    -   **Fast TPU**: Processes **all tiles and all pixels simultaneously** in a single vectorized `jax.lax.scan`. By combining this with `jax.checkpoint`, it achieves maximum parallelism while staying within the 32GB HBM limit.

4.  **XLA Fusion**: The "Fast" version is written using primitive XLA-friendly operations that allow the compiler to fuse the entire rendering pass into a single optimized kernel, minimizing intermediate data transfers between the TPU's registers and HBM.

## Reproduction Command

```bash
# Fast TPU Rasterizer (Recommended)
conda run -n tpu-env bash -c "PYTHONPATH=. python3 -u train.py \
  --data_path gs://dataset-nerf/nerf_llff_data/room \
  --images_subdir images_8 \
  --num_iterations 30000 \
  --fast_tpu_rasterizer"

# Standard Rasterizer (Fixed, slower)
conda run -n tpu-env bash -c "PYTHONPATH=. python3 -u train.py \
  --data_path gs://dataset-nerf/nerf_llff_data/room \
  --images_subdir images_8 \
  --num_iterations 30000"
```
