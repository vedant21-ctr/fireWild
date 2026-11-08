"""
config.py - Load the YAML experiment configuration files in configs/.
"""

import os
from functools import cache

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")


@cache
def load_config(name):
    """Return the parsed contents of configs/<name>.yaml."""
    with open(os.path.join(CONFIG_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)
