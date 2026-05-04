import re

with open("jax_gs/renderer/renderer.py", "r") as f:
    content = f.read()

target = """    # 3. Rasterize tiles
    use_pallas_effective = use_pallas and HAS_PALLAS and backend == "gpu"
    if use_pallas_effective:
        # Rasterize tiles using Pallas
        image, extras = render_tiles_pallas(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background,
            backend=backend
        )
    else:
        # Rasterize tiles using pure JAX
        image = render_tiles(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background
        )
        extras = {}"""

replacement = """    # 3. Rasterize tiles
    if use_pallas and HAS_PALLAS:
        if backend == "tpu":
            from jax_gs.renderer.rasterizer_tpu import render_tiles_tpu
            image = render_tiles_tpu(
                means2D, cov2D, gaussians.opacities, colors,
                sorted_tile_ids, sorted_gaussian_ids,
                camera.H, camera.W, background
            )
            extras = {}
        else:
            # Rasterize tiles using Pallas GPU kernel
            image, extras = render_tiles_pallas(
                means2D, cov2D, gaussians.opacities, colors,
                sorted_tile_ids, sorted_gaussian_ids,
                camera.H, camera.W, TILE_SIZE, background,
                backend=backend
            )
    else:
        # Rasterize tiles using pure JAX standard implementation
        image = render_tiles(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background
        )
        extras = {}"""

new_content = content.replace(target, replacement)
with open("jax_gs/renderer/renderer.py", "w") as f:
    f.write(new_content)
