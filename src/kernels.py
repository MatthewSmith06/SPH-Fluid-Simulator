"""
Smoothing kernel functions.

Used to weight each particle's contribution to a given field at location r,
by distributing its field quantities into the local neighbourhood around it.
"""

import numpy as np
import taichi as ti
from config import *

h = config.h
h2 = config.h2
h3 = config.h3

# 2d normalising constants, precomputed for time efficiency
W_POLY6 = 4 / (np.pi * h ** 8)
gradW_POLY6 = -24 / (np.pi * h ** 8)
lapW_POLY6 = -48 / (np.pi * h ** 8)
W_SPIKY = 10 / (np.pi * h ** 5)
gradW_SPIKY = -30 / (np.pi * h ** 5)
W_VISC = 10 / (3 * np.pi * h ** 2)
lapW_VISC = 40 / (np.pi * h ** 5)

# poly6
@ti.func
def W_poly6(r2):
    return W_POLY6 * (h2 - r2) ** 3

# poly6 gradient
@ti.func
def gradW_poly6(r, r2):
    return gradW_POLY6 * (h2 - r2) ** 2 * r

# poly6 laplacian
@ti.func
def lapW_poly6(r2):
    return lapW_POLY6 * (h2 - r2) * (h2 - 3 * r2)

# spiky
@ti.func
def W_spiky(r):
    return W_SPIKY * (h - r) ** 3

# spiky gradient
@ti.func
def gradW_spiky(r):
    r_len = r.norm() # particle pair separation
    r_unit = r.normalized() # displacement unit vector
    return gradW_SPIKY * (h - r_len) ** 2 * r_unit

# viscosity, unused in the simulation but included for completeness
@ti.func
def W_visc(r, r2):
    r3 = r2 * r # cube of particle pair separation
    return W_VISC * ((-r3 / (2 * h3)) + (r2 / h2) + (h / 2 * r) - 1)

# viscosity laplacian, standard 2d form as the exact laplacian of W_visc is singular at r = 0
@ti.func
def lapW_visc(r):
    r_len = r.norm() # particle pair separation
    return lapW_VISC * (h - r_len)
