# %% [markdown]
# # Exercise 1 — Point clouds: geometry and spread in 2D
#
# Run as a script:   .venv/bin/python docs/exercises/data/code/ex1.py
# Or interactively:  Shift+Enter on any `# %%` cell (VS Code Interactive Window).

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


def save(fig, name):
    """Write a figure to figures/<name>.png."""
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  wrote {path.name}")


# %% [markdown]
# ## A — Generate the clouds

# %%
def make_clouds(scale=1.0):
    """Return (X, y) for the 4 classes with all stds multiplied by `scale`.

    X is (400, 2), y is (400,) of ints 0..3.

    TODO: decide independent draws vs. common random numbers (x = mu + s*sigma*z)
    and document the choice in the report.
    """
    raise NotImplementedError


def figure_1(X, y):
    """Scatter of all 400 points, one colour per class, centres marked.

    TODO: title, axis labels, class legend — all three are graded.
    """
    raise NotImplementedError


# %% [markdown]
# ## B — More or less spread out

# %%
def figure_2(datasets):
    """4 subplots, one per scale, sharing axis limits (rubric: shared axes)."""
    raise NotImplementedError


def separation_ratios():
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j) for the 6 pairs.

    Purely geometric — needs no sampled data. Return something you can print
    as a table and take the min of.
    """
    raise NotImplementedError


def mixing_rate(X, y):
    """Fraction of points whose nearest *design* centre is not their own class.

    Hint: X[:, None, :] - MEANS[None, :, :] -> (400, 4, 2), then norm over
    axis=2, then argmin over axis=1.
    """
    raise NotImplementedError


def figure_3(rates):
    """Mixing rate vs. scale factor s."""
    raise NotImplementedError


# %% [markdown]
# ## Run everything
#
# Every number printed here goes into the report text and the Results summary.

# %%
def main():
    print("Exercise 1")
    # TODO: build the s=1 dataset, draw Figure 1
    # TODO: build all four datasets, draw Figure 2
    # TODO: print the r_ij table, name the smallest pair, state its value at s=2
    # TODO: print the mixing rate for each s, draw Figure 3


if __name__ == "__main__":
    main()
