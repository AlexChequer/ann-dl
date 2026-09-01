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
    code_dir = Path(__file__).resolve().parent
except NameError:
    repo_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "mkdocs.yml").exists())
    code_dir = repo_root / "docs" / "exercises" / "data" / "code"

figures_dir = code_dir.parent / "figures"
figures_dir.mkdir(exist_ok=True)

# Class parameters from item A. Row k is class k, column 0 is x and column 1 is y.
CLASS_MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
CLASS_STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])

N_CLASSES = 4
N_DIMENSIONS = 2
SAMPLES_PER_CLASS = 100
SCALE_FACTORS = [0.5, 1.0, 2.0, 4.0]

# %% [markdown]
# ## A — Generate the clouds

# %%
# Draw the cloud "shapes" once: standard_shapes[k] holds class k's 100 points,
# each drawn from a standard normal. Reusing the same shapes across every scale
# factor is the common-random-numbers choice — Figure 2's panels then differ
# only because of s, never because the data was resampled. (For independent
# draws instead, move this line inside generate_clouds so each call redraws.)
standard_shapes = rng.standard_normal((N_CLASSES, SAMPLES_PER_CLASS, N_DIMENSIONS))


def generate_clouds(scale=1.0):
    """Return (points, labels) for the 4 classes, every std multiplied by `scale`.

    A standard normal z becomes a point of class k via x = mean_k + s * std_k * z.
    CLASS_MEANS[k] and CLASS_STDS[k] are length-2 while standard_shapes[k] is
    (100, 2), so NumPy broadcasts the parameters down the columns: column 0 gets
    the x parameters, column 1 the y.
    """
    points = np.vstack([
        CLASS_MEANS[class_index] + scale * CLASS_STDS[class_index] * standard_shapes[class_index]
        for class_index in range(N_CLASSES)
    ])
    labels = np.repeat(np.arange(N_CLASSES), SAMPLES_PER_CLASS)
    return points, labels


points, labels = generate_clouds(scale=1.0)

# Sanity check: shapes, class counts, and empirical vs. design parameters.
print("points", points.shape, " labels", labels.shape, " counts", np.bincount(labels))
print(f"{'class':<6} {'empirical mean':<20} {'design mean':<16} {'empirical std':<20} {'design std'}")
for class_index in range(N_CLASSES):
    in_class = points[labels == class_index]
    print(
        f"{class_index:<6} {str(in_class.mean(axis=0).round(2)):<20} "
        f"{str(CLASS_MEANS[class_index]):<16} {str(in_class.std(axis=0).round(2)):<20} "
        f"{CLASS_STDS[class_index]}"
    )
