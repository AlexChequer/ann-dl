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

# %% [markdown]
# ## A — Generate the clouds

# %%
# Draw the cloud "shapes" once: Z[k] holds class k's 100 standard-normal points.
# Reusing the same Z across every scale factor is the common-random-numbers
# choice — Figure 2's panels then differ only because of s, never because the
# data was resampled. (For independent draws instead, move this line inside
# make_clouds so each call redraws.)
Z = rng.standard_normal((4, N_PER_CLASS, 2))


def make_clouds(scale=1.0):
    """Return (X, y) for the 4 classes with every std multiplied by `scale`.

    A standard normal z is turned into a point of class k by x = mu_k + s*sigma_k*z.
    MEANS[k] and STDS[k] are length-2, Z[k] is (100, 2), so NumPy broadcasts the
    parameters down the columns: column 0 gets the x parameters, column 1 the y.
    """
    X = np.vstack([MEANS[k] + scale * STDS[k] * Z[k] for k in range(4)])
    y = np.repeat(np.arange(4), N_PER_CLASS)
    return X, y


X, y = make_clouds(1.0)

# Sanity check: shapes, class counts, and empirical vs. design parameters.
print("X", X.shape, " y", y.shape, " counts", np.bincount(y))
print(f"{'k':<3} {'empirical mean':<20} {'design mean':<16} {'empirical std':<20} {'design std'}")
for k in range(4):
    print(
        f"{k:<3} {str(X[y == k].mean(axis=0).round(2)):<20} {str(MEANS[k]):<16} "
        f"{str(X[y == k].std(axis=0).round(2)):<20} {STDS[k]}"
    )
