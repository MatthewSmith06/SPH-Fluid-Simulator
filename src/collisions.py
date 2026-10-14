"""
Collision functions.
"""

import taichi as ti
from fields import *
from config import *

@ti.func
def particle_collision(i, j, r, r2):
    if r2 <= (2 * config.radius) ** 2:
        dist = r.norm()
        n_ = r / dist # collision normal

        # velocity components along the normal
        vi_n = vel[i].dot(n_)
        vj_n = vel[j].dot(n_)

        rel_n = vi_n - vj_n # relative normal velocity
        if rel_n < 0: # only resolve if approaching
            imp = -(1 + config.e) * rel_n / 2.0 # impulse, assuming equal masses
            vel[i] += imp * n_
            vel[j] -= imp * n_

        # push particles apart to prevent overlap
        overlap = 2 * config.radius - dist
        pos[i] += 0.5 * overlap * n_
        pos[j] -= 0.5 * overlap * n_

@ti.func
def handle_boundary_collisions():
    min_bound = config.radius
    max_bound = 1 - min_bound

    for i in range(n):
        for d in ti.static([0, 1]): # d=0 <-> x, d=1 <-> y
            if pos[i][d] < min_bound:
                pos[i][d] = min_bound
                vel[i][d] = -vel[i][d] * config.e
            elif pos[i][d] > max_bound:
                pos[i][d] = max_bound
                vel[i][d] = -vel[i][d] * config.e
