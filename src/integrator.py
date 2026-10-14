"""
Integration functions.
"""

import taichi as ti
from fields import *

@ti.func
def clamp_acceleration(i):
    amax = 1e3
    a = acc[i].norm()

    if a > amax:
        acc[i] = acc[i] / a * amax

# semi-implicit Euler integration
@ti.func
def semi_implicit_euler():
    for i in range(n):
        acc[i] = (f_pres[i] + f_visc[i] + f_surf[i]) / rho[i] + config.gravity

        clamp_acceleration(i)

        vel[i] += acc[i] * config.dt
        pos[i] += vel[i] * config.dt

# TODO: add other integration methods
