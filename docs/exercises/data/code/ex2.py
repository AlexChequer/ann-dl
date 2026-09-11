# %% [markdown]
# # Exercise 2 — Non-linearity in higher dimensions

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)

try:
    code_dir = Path(__file__).resolve().parent
except NameError:
    repo_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "mkdocs.yml").exists())
    code_dir = repo_root / "docs" / "exercises" / "data" / "code"

figures_dir = code_dir.parent / "figures"
figures_dir.mkdir(exist_ok=True)

SAMPLES_PER_CLASS = 500
N_DIMENSIONS = 5
PAIR_COLORS = ["#4C72B0", "#C44E52"]

# %% [markdown]
# ## A — Dataset I: shifted Gaussians

# %%
MEAN_A = np.zeros(N_DIMENSIONS)
COV_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

MEAN_B = np.full(N_DIMENSIONS, 1.5)
COV_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

class_a = rng.multivariate_normal(MEAN_A, COV_A, size=SAMPLES_PER_CLASS)
class_b = rng.multivariate_normal(MEAN_B, COV_B, size=SAMPLES_PER_CLASS)

dataset_i = np.vstack([class_a, class_b])
labels_i = np.repeat([0, 1], SAMPLES_PER_CLASS)

# %% [markdown]
# ## B — Dataset II: concentric shells

# %%
# The statement writes the radius as N(2.0, 0.4) and N(5.0, 0.4); 0.4 is read
# as the standard deviation, which is what np.random.normal's `scale` expects.
CORE_RADIUS_MEAN, SHELL_RADIUS_MEAN, RADIUS_STD = 2.0, 5.0, 0.4


def sample_shell(radius_mean, n_samples):
    """Points at radius ~ N(radius_mean, 0.4) in a uniformly random direction."""
    directions = rng.standard_normal((n_samples, N_DIMENSIONS))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.normal(radius_mean, RADIUS_STD, size=(n_samples, 1))
    return radii * directions


class_c = sample_shell(CORE_RADIUS_MEAN, SAMPLES_PER_CLASS)
class_d = sample_shell(SHELL_RADIUS_MEAN, SAMPLES_PER_CLASS)

dataset_ii = np.vstack([class_c, class_d])
labels_ii = np.repeat([0, 1], SAMPLES_PER_CLASS)

# %% [markdown]
# ## C — Visualize and compare

# %%
def project_to_2d(points):
    """PCA down to 2 components; returns the projection and its explained variance."""
    pca = PCA(n_components=2)
    projected = pca.fit_transform(points)
    return projected, pca.explained_variance_ratio_


projection_i, variance_i = project_to_2d(dataset_i)
projection_ii, variance_ii = project_to_2d(dataset_ii)

PANELS = [
    ("Dataset I — shifted Gaussians", projection_i, labels_i, variance_i, ("A", "B")),
    ("Dataset II — concentric shells", projection_ii, labels_ii, variance_ii, ("C", "D")),
]


def plot_figure_4():
    """Figure 4 — both datasets projected to 2D by PCA, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (title, projected, labels, variance, names) in zip(axes, PANELS):
        for index, name in enumerate(names):
            in_class = projected[labels == index]
            ax.scatter(in_class[:, 0], in_class[:, 1], s=12, alpha=0.55,
                       color=PAIR_COLORS[index], label=f"Class {name}")
        ax.set_title(f"{title}\nPC1 + PC2 = {variance.sum():.1%} of variance")
        ax.set_xlabel(f"PC1 ({variance[0]:.1%})")
        ax.set_ylabel(f"PC2 ({variance[1]:.1%})")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle("Figure 4 — PCA projection to 2D")
    fig.savefig(figures_dir / "fig4.png", dpi=150, bbox_inches="tight")
    return fig


def centre_distance(points, labels):
    """Distance between the two class means, measured in the original 5D space."""
    return float(np.linalg.norm(points[labels == 0].mean(axis=0) - points[labels == 1].mean(axis=0)))


def plot_figure_5():
    """Figure 5 — histogram of the radius ||x|| of each point, classes overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (title, _, labels, _, names) in zip(axes, PANELS):
        points = dataset_i if names == ("A", "B") else dataset_ii
        radii = np.linalg.norm(points, axis=1)
        for index, name in enumerate(names):
            ax.hist(radii[labels == index], bins=40, alpha=0.6,
                    color=PAIR_COLORS[index], label=f"Class {name}")
        ax.set_title(title.split(" — ")[0])
        ax.set_xlabel(r"radius $\|x\|$")
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle(r"Figure 5 — distribution of the radius $\|x\|$ in 5D")
    fig.savefig(figures_dir / "fig5.png", dpi=150, bbox_inches="tight")
    return fig


distance_i = centre_distance(dataset_i, labels_i)
distance_ii = centre_distance(dataset_ii, labels_ii)

plot_figure_4()
plot_figure_5()

print(f"Dataset I   centre distance in 5D: {distance_i:.3f}")
print(f"Dataset II  centre distance in 5D: {distance_ii:.3f}")
print(f"Dataset I   PC1 {variance_i[0]:.1%} + PC2 {variance_i[1]:.1%} = {variance_i.sum():.1%}")
print(f"Dataset II  PC1 {variance_ii[0]:.1%} + PC2 {variance_ii[1]:.1%} = {variance_ii.sum():.1%}")

# %% [markdown]
# ## D — Analysis

# %%
# The separating function proposed in the analysis: squared norm against a threshold.
SQUARED_RADIUS_THRESHOLD = ((CORE_RADIUS_MEAN + SHELL_RADIUS_MEAN) / 2) ** 2


def classify_by_squared_norm(points):
    """Class C (0) if ||x||^2 is below the threshold, class D (1) otherwise."""
    return (np.sum(points**2, axis=1) > SQUARED_RADIUS_THRESHOLD).astype(int)


accuracy = float((classify_by_squared_norm(dataset_ii) == labels_ii).mean())
print(f"\n||x||^2 > {SQUARED_RADIUS_THRESHOLD:.2f} separates Dataset II with accuracy {accuracy:.1%}")
