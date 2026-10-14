"""
Fields.

Used to store each particle's quantities, representing the simulation state.
"""

import taichi as ti
from config import *
import kernels as W

n = config.n

# scalar fields
mass = ti.field(dtype=ti.f32, shape=n)
rho = ti.field(dtype=ti.f32, shape=n) # density
pres = ti.field(dtype=ti.f32, shape=n) # pressure
lap_col = ti.field(dtype=ti.f32, shape=n) # colour laplacian

# vector fields
pos = ti.Vector.field(2, dtype=ti.f32, shape=n) # pos is normalised between 0 to 1
vel = ti.Vector.field(2, dtype=ti.f32, shape=n)
acc = ti.Vector.field(2, dtype=ti.f32, shape=n)
grad_col = ti.Vector.field(2, dtype=ti.f32, shape=n) # colour gradient
f_pres = ti.Vector.field(2, dtype=ti.f32, shape=n) # pressure force
f_visc = ti.Vector.field(2, dtype=ti.f32, shape=n) # viscosity force
f_surf = ti.Vector.field(2, dtype=ti.f32, shape=n) # surface tension force

@ti.func
def reset():
    for i in range(n):
        rho[i] = mass[i] * W.W_spiky(0.0) # self-contribution, as the neighbour searches skip j = i
        lap_col[i] = 0.0
        grad_col[i] = ti.Vector([0.0, 0.0])
        f_pres[i] = ti.Vector([0.0, 0.0])
        f_visc[i] = ti.Vector([0.0, 0.0])
        f_surf[i] = ti.Vector([0.0, 0.0])
