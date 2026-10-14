"""
Physics functions.
"""

import taichi as ti
from config import *
from fields import *
import kernels as W

rho0 = config.rho0
k = config.k

@ti.func
def density(i, j, r, r2):
    rho[i] += mass[j] * W.W_spiky(ti.sqrt(r2)) # same kernel as the pressure gradient, so energy is conserved
    rho[j] += mass[i] * W.W_spiky(ti.sqrt(r2))

@ti.func
def pressure(i):
    pres[i] = max(0.0, k * (rho[i] - rho0)) # avoid negative pressure

@ti.func
def pressure_force(i, j, r, r2):
    f_temp = mass[j] * ((pres[i] + pres[j]) / (2 * rho[j])) * W.gradW_spiky(r) # symmetrise force using arithmetic mean of pressures of interacting particles
    f_pres[i] -= f_temp
    f_pres[j] += f_temp

@ti.func
def viscosity_force(i, j, r, r2):
    f_temp = config.mu * mass[j] * ((vel[j] - vel[i]) / rho[j]) * W.lapW_visc(r) # symmetrise force using velocity difference of interacting particles
    f_visc[i] += f_temp
    f_visc[j] -= f_temp

@ti.func
def colour_gradient(i, j, r, r2):
    grad_col[i] += mass[j] * 1 / rho[j] * W.gradW_poly6(r, r2)
    grad_col[j] -= mass[i] * 1 / rho[i] * W.gradW_poly6(r, r2)

@ti.func
def colour_laplacian(i, j, r, r2):
    lap_col[i] += mass[j] * 1 / rho[j] * W.lapW_poly6(r2)
    lap_col[j] += mass[i] * 1 / rho[i] * W.lapW_poly6(r2)

@ti.func
def surface_tension(i):
    if grad_col[i].norm() > config.surface_threshold:
        f_surf[i] = - config.sigma * lap_col[i] * grad_col[i] / grad_col[i].norm()
    else:
        f_surf[i] = ti.Vector([0.0, 0.0])
