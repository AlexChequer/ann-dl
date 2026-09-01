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


# %%
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_figure_1(points, labels):
    """Figure 1 — the four clouds at s = 1, with each class centre marked."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for class_index in range(N_CLASSES):
        in_class = points[labels == class_index]
        ax.scatter(
            in_class[:, 0], in_class[:, 1],
            s=18, alpha=0.6, color=CLASS_COLORS[class_index],
            label=f"Class {class_index}",
        )

    ax.scatter(
        CLASS_MEANS[:, 0], CLASS_MEANS[:, 1],
        marker="X", s=220, c=CLASS_COLORS, edgecolors="black", linewidths=1.5,
        zorder=3, label="Class centres",
    )

    ax.set_title("Figure 1 — Four Gaussian point clouds in 2D ($s = 1$)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.2)

    fig.savefig(figures_dir / "fig1.png", dpi=150, bbox_inches="tight")
    return fig


plot_figure_1(points, labels)
