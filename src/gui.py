"""
GUI functions.

Used to render the particles by splatting each one onto an image.
"""

import taichi as ti
from fields import *

SPLAT_RADIUS = 12

image = ti.Vector.field(3, dtype=ti.f32, shape=(1200, 800))

@ti.kernel
def clear_image():
    for I in ti.grouped(image):
        image[I] = 0.0

@ti.kernel
def splat():
    for i in pos:
        cx = int(pos[i].x * 1200)
        cy = int(pos[i].y * 800)

        for dx, dy in ti.ndrange(
            (-SPLAT_RADIUS, SPLAT_RADIUS + 1),
            (-SPLAT_RADIUS, SPLAT_RADIUS + 1)
        ):
            x = cx + dx
            y = cy + dy

            if 0 <= x < 1200 and 0 <= y < 800:
                r2 = dx * dx + dy * dy

                if r2 < SPLAT_RADIUS * SPLAT_RADIUS:
                    w = ti.exp(-r2 / 200.0)

                    # rgb contribution
                    ti.atomic_add(image[x, y][0], config.rgb[0] * w)
                    ti.atomic_add(image[x, y][1], config.rgb[1] * w)
                    ti.atomic_add(image[x, y][2], config.rgb[2] * w)

@ti.kernel
def normalize_image():
    for I in ti.grouped(image):
        image[I] = 1.0 - ti.exp(-image[I] * 0.5)
