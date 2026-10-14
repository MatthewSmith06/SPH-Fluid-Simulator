"""
Load and store constants
"""

import os
from pathlib import Path

import taichi as ti
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIG_PATH = os.environ.get("SPH_CONFIG", CONFIG_DIR / "splash_tank.yaml")

class Config:
    def __init__(self):
        with open(CONFIG_PATH, "r") as f:
            config_dict = yaml.safe_load(f)

        for key, value in config_dict.items():
            setattr(self, key, value)

        self.h2 = self.h * self.h
        self.h3 = self.h2 * self.h

        self.gravity = ti.Vector([self.gravity_x, self.gravity_y])
        self.initial_velocity = ti.Vector([self.initial_velocity_x, self.initial_velocity_y])

config = Config()