# %% [markdown]
# # Exercise 1 — Point clouds: geometry and spread in 2D

# %%
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# One generator for the whole report, as the statement requires.
rng = np.random.default_rng(42)

# Resolve paths whether this runs as a script (__file__ exists) or in a
# Jupyter kernel (it does not) — walk up to the repo root, marked by mkdocs.yml.
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "mkdocs.yml").exists())
    HERE = root / "docs" / "exercises" / "data" / "code"

FIGURES = HERE.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# Class parameters from item A. Row k is class k.
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
SCALES = [0.5, 1.0, 2.0, 4.0]
