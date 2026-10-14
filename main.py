"""
Main.

Used to run the simulation, or to time it over a set number of frames.
"""

import sys
import time
from pathlib import Path
import taichi as ti
ti.init(arch=ti.gpu) # must run before the other imports as they depend on taichi being initialised (ti.gpu is CUDA for windows, metal for mac)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src")) # so the modules in src can be imported outside PyCharm

from config import *
from fields import *
from kernels import *
from physics import *
from integrator import *
from neighbours import *
from solver import *
from gui import *

# TODO: replace import * with import ... as ...
# TODO: restructure modules so we have gui, pressure, density, forces, integrator, collisions, solver, neighbours (containing the update generalisation with spatial hashing)

def run():
    init()
    gui = ti.GUI("SPH", (1200, 800))
    while gui.running:
        for _ in range(2): # run multiple small steps before GUI
            step_spatially()

        clear_image()
        splat()
        normalize_image()

        gui.set_image(image)
        gui.show()

def run_n_frames_naive(num_frames):
    start = time.perf_counter()
    init()
    gui = ti.GUI("SPH", (1200, 800))

    for _ in range(num_frames):
        step()

        clear_image()
        splat()
        normalize_image()

        gui.set_image(image)
        gui.show()

    ti.sync() # ensure all queued work has finished
    elapsed = time.perf_counter() - start
    fps = num_frames / elapsed

    return fps

def run_n_frames_optimised(num_frames):
    start = time.perf_counter()
    init()
    gui = ti.GUI("SPH", (1200, 800))

    for _ in range(num_frames):
        step_spatially()

        clear_image()
        splat()
        normalize_image()

        gui.set_image(image)
        gui.show()

    ti.sync() # ensure all queued work has finished
    elapsed = time.perf_counter() - start
    fps = num_frames / elapsed

    return fps

if __name__ == "__main__":
    run()
