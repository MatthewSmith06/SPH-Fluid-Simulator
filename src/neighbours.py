"""
Neighbour search functions.

Used to apply an update to each particle, or each particle pair within the
smoothing radius, either naively or using a spatial lookup grid.
"""

import taichi as ti
import numpy as np
from config import *
from fields import *

h2 = config.h2
r2_min = 1e-12 # minimum squared separation, skips coincident particles to avoid division by zero

cell_size = config.h
grid_width = int(np.ceil(1.0 / cell_size))
grid_height = int(np.ceil(1.0 / cell_size))
spatial_lookup_grid = ti.field(dtype=ti.i32, shape=(grid_width, grid_height, n))
cell_counts = ti.field(dtype=ti.i32, shape=(grid_width, grid_height))

@ti.func
def pos_to_cell_coord(i):
    cell_x = int(pos[i][0] / cell_size)
    cell_y = int(pos[i][1] / cell_size)

    # clamp cell coordinates so particles pushed outside the grid go to the nearest edge cell
    cell_x = ti.math.clamp(cell_x, 0, grid_width - 1)
    cell_y = ti.math.clamp(cell_y, 0, grid_height - 1)

    return ti.Vector([cell_x, cell_y])

@ti.func
def update_spatial_lookup_grid():
    # reset cell_counts
    for i in range(grid_width):
        for j in range(grid_height):
            cell_counts[i, j] = 0

    # create spatial lookup
    for i in range(n):
        cell_coord = pos_to_cell_coord(i)
        coord_x = cell_coord[0]
        coord_y = cell_coord[1]

        index = ti.atomic_add(cell_counts[coord_x, coord_y], 1) # handle race condition by tracking how many particles want to write to given cell at same time
        spatial_lookup_grid[coord_x, coord_y, index] = i

    # TODO: make the hash function use mod so we don't need a nested loop

@ti.func
def update_neighbourwise(field: ti.template()):
    for i in range(n):
        cell_coord = pos_to_cell_coord(i) # TODO: don't call twice, optimise
        coord_x = cell_coord[0]
        coord_y = cell_coord[1]

        for a in range(3):
            for b in range(3):
                if 0 <= coord_x + (a - 1) < grid_width and 0 <= coord_y + (b - 1) < grid_height:
                    for j in range(cell_counts[coord_x + (a - 1), coord_y + (b - 1)]):
                        neighbour = spatial_lookup_grid[coord_x + (a - 1), coord_y + (b - 1), j]
                        if neighbour > i: # don't double count
                            r = pos[i] - pos[neighbour] # displacement vector
                            r2 = r.dot(r) # square of particle pair separation
                            if r2_min < r2 < h2:
                                field(i, neighbour, r, r2)

"""
# hash function
@ti.func
def cell_coord_to_hash_value(cell_x, cell_y):
    prime1 = 15823 # make new ones yourself
    prime2 = 9737333
    x = cell_x * prime1
    y = cell_y * prime2
    return x + y
"""

"""
@ti.func
def hash_value_to_cell_key(hash_value):
    return hash_value % n
"""

@ti.func
def update_pairwise(field: ti.template()):
    for i in range(n):
        for j in range(i + 1, n):
            r = pos[i] - pos[j] # displacement vector
            r2 = r.dot(r) # square of particle pair separation
            if r2_min < r2 < h2:
                field(i, j, r, r2)

@ti.func
def update_particlewise(field: ti.template()):
    for i in range(n):
        field(i)
