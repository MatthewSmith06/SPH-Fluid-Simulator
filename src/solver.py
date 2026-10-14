"""
Solver functions.

Used to initialise the simulation and advance it by one time step.
"""

import taichi as ti
import numpy as np
from config import *
from fields import *
from physics import *
from integrator import *
from collisions import *
from neighbours import *

n = config.n
h2 = config.h2
side = int(np.ceil(np.sqrt(n))) # particles per row of the initial grid

# initialisation
@ti.kernel
def init():
    for i in range(n):
        pos[i] = ti.Vector([0.1 + 0.8 * (i % side + 0.5) / side, 0.1 + 0.8 * (i // side + 0.5) / side]) # regular grid, as random positions store compression energy
        vel[i] = config.initial_velocity
        acc[i] = config.gravity
        mass[i] = config.rest_mass
        rho[i] = config.rho0

@ti.kernel
def step():
    reset()

    update_pairwise(density)
    update_particlewise(pressure)

    update_pairwise(grouped_interactions)
    update_particlewise(surface_tension)

    semi_implicit_euler()

    handle_boundary_collisions()
    update_pairwise(particle_collision) # handle particle collisions after boundary collisions because boundary collision handling will place some particles in the same position (e.g: particles with x<0 and y<0)
    handle_boundary_collisions() # particle collisions can push particles back outside the boundary

@ti.kernel
def step_spatially():
    reset()
    update_spatial_lookup_grid()

    update_neighbourwise(density)
    update_particlewise(pressure)

    update_neighbourwise(grouped_interactions)
    update_particlewise(surface_tension)

    semi_implicit_euler()

    handle_boundary_collisions()
    update_spatial_lookup_grid() # rebuild as particles have moved since the start of the step
    update_neighbourwise(particle_collision) # handle particle collisions after boundary collisions because boundary collision handling will place some particles in the same position (e.g: particles with x<0 and y<0)
    handle_boundary_collisions() # particle collisions can push particles back outside the boundary

# group pressure force, viscosity force, colour gradient and colour laplacian into a single neighbour search
@ti.func
def grouped_interactions(i, j, r, r2):
    pressure_force(i, j, r, r2)
    viscosity_force(i, j, r, r2)
    colour_gradient(i, j, r, r2)
    colour_laplacian(i, j, r, r2)
