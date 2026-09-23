# %% [markdown]
# # Perceptron — separable vs. overlapping data

# %%
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

# One generator for the whole report. Every random draw below — data and
# weight initialisation alike — comes from it, in the order the cells run.
rng = np.random.default_rng(42)

# Resolve paths whether this runs as a script (__file__ exists) or in a
# Jupyter kernel (it does not) — walk up to the repo root, marked by mkdocs.yml.
try:
    code_dir = Path(__file__).resolve().parent
except NameError:
    repo_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "mkdocs.yml").exists())
    code_dir = repo_root / "docs" / "exercises" / "perceptron" / "code"

figures_dir = code_dir.parent / "figures"
figures_dir.mkdir(exist_ok=True)

SAMPLES_PER_CLASS = 1000
CLASS_COLORS = ["#4C72B0", "#C44E52"]


def generate_two_classes(means, covariance):
    """Draw SAMPLES_PER_CLASS points per class from N(mean, covariance).

    Returns the points stacked class 0 first, then class 1, and the 0/1 labels.
    """
    points = np.vstack([
        rng.multivariate_normal(mean, covariance, size=SAMPLES_PER_CLASS)
        for mean in means
    ])
    labels = np.repeat([0, 1], SAMPLES_PER_CLASS)
    return points, labels


def scatter_classes(ax, points, labels):
    """Scatter both classes on ax, one colour each (reusable under a decision boundary)."""
    for class_index in (0, 1):
        in_class = points[labels == class_index]
        ax.scatter(in_class[:, 0], in_class[:, 1], s=10, alpha=0.5,
                   color=CLASS_COLORS[class_index], label=f"Class {class_index}")


def plot_dataset(points, labels, title, filename):
    """Scatter plot of a two-class dataset, saved under figures/."""
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, points, labels)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")  # keep angles true, so a boundary drawn later looks as it is
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    return fig


# %% [markdown]
# ## Exercise 1 — Separable data
#
# ### A — Generate the data

# %%
EX1_MEANS = [np.array([1.5, 1.5]), np.array([5.0, 5.0])]
EX1_COVARIANCE = np.array([[0.5, 0.0], [0.0, 0.5]])  # shared by both classes

ex1_points, ex1_labels = generate_two_classes(EX1_MEANS, EX1_COVARIANCE)

plot_dataset(ex1_points, ex1_labels,
             "Figure 1 — Exercise 1: two separable Gaussian classes", "fig1.png")

# %% [markdown]
# ### B — Implement the perceptron

# %%
LEARNING_RATE = 0.01
MAX_EPOCHS = 100


def step(score):
    """The activation of the statement: 1 when the score is >= 0, and 0 otherwise."""
    return 1 if score >= 0 else 0


def predict(points, weights, bias):
    """step(w · x + b) applied to every row of points at once."""
    return (points @ weights + bias >= 0).astype(int)


def accuracy(points, labels, weights, bias):
    """Fraction of the full dataset that the weights classify correctly."""
    return float((predict(points, weights, bias) == labels).mean())


class TrainingRun(NamedTuple):
    """Everything one training run has to report."""

    weights: np.ndarray
    bias: float
    epochs_run: int
    accuracy_history: list
    update_history: list
    pocket_weights: np.ndarray
    pocket_bias: float
    pocket_accuracy: float
    pocket_epoch: int
    pocket_accuracy_history: list


def train_perceptron(points, labels, initial_weights,
                     learning_rate=LEARNING_RATE, max_epochs=MAX_EPOCHS):
    """Train the perceptron sample by sample, keeping the best weights in a pocket.

    One sample at a time: predict, and on a mistake move the weights by
    learning_rate * (label - prediction) * point. A correct prediction gives an
    error of 0 and therefore no update, so an epoch that updates nothing has
    separated the data and training stops.

    The pocket is the only addition to that loop: whenever an update leaves the
    weights more accurate on the full dataset than anything seen so far, they are
    set aside. Nothing is ever modified in place — every update rebinds `weights`
    to a new array — so the pocket can simply keep a reference to the old one.
    """
    weights, bias = initial_weights, 0.0
    pocket_weights, pocket_bias = weights, bias
    pocket_accuracy = accuracy(points, labels, weights, bias)
    pocket_epoch = 0

    accuracy_history, pocket_accuracy_history, update_history = [], [], []
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        updates = 0

        for point, label in zip(points, labels):
            error = label - step(point @ weights + bias)
            if error == 0:
                continue

            weights = weights + learning_rate * error * point
            bias = bias + learning_rate * error
            updates += 1

            current_accuracy = accuracy(points, labels, weights, bias)
            if current_accuracy > pocket_accuracy:
                pocket_weights, pocket_bias = weights, bias
                pocket_accuracy, pocket_epoch = current_accuracy, epoch

        epochs_run = epoch
        update_history.append(updates)
        accuracy_history.append(accuracy(points, labels, weights, bias))
        pocket_accuracy_history.append(pocket_accuracy)

        if updates == 0:  # a full pass with nothing to correct: converged
            break

    return TrainingRun(weights, bias, epochs_run, accuracy_history, update_history,
                       pocket_weights, pocket_bias, pocket_accuracy, pocket_epoch,
                       pocket_accuracy_history)


def plot_boundary(ax, weights, bias, points, color, label):
    """Draw the line w · x + b = 0 across the horizontal range of the data."""
    x_values = np.array([points[:, 0].min(), points[:, 0].max()])
    # w1 * x1 + w2 * x2 + b = 0  =>  x2 = -(w1 * x1 + b) / w2
    ax.plot(x_values, -(weights[0] * x_values + bias) / weights[1],
            color=color, linewidth=2, label=label)


def plot_decision_figure(points, labels, boundaries, misclassified, title, filename):
    """Data, one or more decision boundaries, and the misclassified points ringed."""
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, points, labels)
    ax.scatter(points[misclassified, 0], points[misclassified, 1],
               s=30, facecolors="none", edgecolors="black", linewidths=0.6,
               label=f"Misclassified ({misclassified.sum()})")

    for weights, bias, color, label in boundaries:
        plot_boundary(ax, weights, bias, points, color, label)

    # Fix the view to the data, so a near-vertical boundary cannot stretch the axes.
    ax.set_xlim(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5)
    ax.set_ylim(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_accuracy_figure(curves, title, filename, legend_loc="lower right"):
    """Accuracy against epoch, one line per curve."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for values, color, label in curves:
        ax.plot(range(1, len(values) + 1), values,
                color=color, marker="o", markersize=3, label=label)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy on the full dataset")
    ax.legend(loc=legend_loc, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    return fig


# Non-zero start, as item B requires. Drawn once and reused by the eta = 1.0 run
# of item D, which must differ from this one in the learning rate alone.
ex1_initial_weights = rng.normal(0, 0.01, size=2)
print("initial w:", ex1_initial_weights, " initial b: 0.0")

# %% [markdown]
# ### C — Train and measure

# %%
ex1_run = train_perceptron(ex1_points, ex1_labels, ex1_initial_weights)

print(f"final w: {ex1_run.weights}")
print(f"final b: {ex1_run.bias:.4f}")
print(f"epochs:  {ex1_run.epochs_run}")
print(f"final accuracy: {ex1_run.accuracy_history[-1]:.4f}")
print(f"accuracy per epoch: {[round(value, 4) for value in ex1_run.accuracy_history]}")
print(f"updates per epoch:  {ex1_run.update_history}")

# %%
ex1_misclassified = predict(ex1_points, ex1_run.weights, ex1_run.bias) != ex1_labels

plot_decision_figure(
    ex1_points, ex1_labels,
    [(ex1_run.weights, ex1_run.bias, "black", "Decision boundary")],
    ex1_misclassified,
    "Figure 2 — Exercise 1: decision boundary after training", "fig2.png",
)

plot_accuracy_figure(
    [(ex1_run.accuracy_history, "#4C72B0", "Accuracy")],
    "Figure 3 — Exercise 1: accuracy per epoch ($\\eta = 0.01$)", "fig3.png",
)

# %% [markdown]
# ### D — Analysis

# %%
# Same data, same starting weights, same everything except the learning rate.
ex1_fast_run = train_perceptron(ex1_points, ex1_labels, ex1_initial_weights, learning_rate=1.0)

print(f"eta = 1.00 -> w: {ex1_fast_run.weights}, b: {ex1_fast_run.bias:.4f}, "
      f"epochs: {ex1_fast_run.epochs_run}, accuracy: {ex1_fast_run.accuracy_history[-1]:.4f}")

slow_direction = ex1_run.weights / np.linalg.norm(ex1_run.weights)
fast_direction = ex1_fast_run.weights / np.linalg.norm(ex1_fast_run.weights)
angle = np.degrees(np.arccos(np.clip(slow_direction @ fast_direction, -1.0, 1.0)))

print(f"w/||w||  at eta = 0.01: {slow_direction}")
print(f"w/||w||  at eta = 1.00: {fast_direction}")
print(f"angle between them: {angle:.2f} degrees")
print(f"||w|| at eta = 0.01: {np.linalg.norm(ex1_run.weights):.4f}, "
      f"at eta = 1.00: {np.linalg.norm(ex1_fast_run.weights):.4f}")

# |b| / ||w|| is how far the boundary sits from the origin: the offset the bias
# has to accumulate before the line can reach the gap between the two clouds.
for name, run in [("eta = 0.01", ex1_run), ("eta = 1.00", ex1_fast_run)]:
    print(f"{name}: {sum(run.update_history)} updates in total, "
          f"|b| / ||w|| = {abs(run.bias) / np.linalg.norm(run.weights):.3f}")
print(f"distance from the origin to the midpoint between the clouds: "
      f"{np.linalg.norm((EX1_MEANS[0] + EX1_MEANS[1]) / 2):.3f}")

# %% [markdown]
# ## Exercise 2 — Overlapping data
#
# ### A — Generate the data

# %%
EX2_MEANS = [np.array([3.0, 3.0]), np.array([4.0, 4.0])]
EX2_COVARIANCE = np.array([[1.5, 0.0], [0.0, 1.5]])  # shared by both classes

ex2_points, ex2_labels = generate_two_classes(EX2_MEANS, EX2_COVARIANCE)

plot_dataset(ex2_points, ex2_labels,
             "Figure 4 — Exercise 2: two overlapping Gaussian classes", "fig4.png")

# %% [markdown]
# ### B — Train, keeping the best weights

# %%
ex2_initial_weights = rng.normal(0, 0.01, size=2)
ex2_run = train_perceptron(ex2_points, ex2_labels, ex2_initial_weights)

print(f"initial w: {ex2_initial_weights}")
print(f"final  w: {ex2_run.weights}, b: {ex2_run.bias:.4f}, "
      f"accuracy: {ex2_run.accuracy_history[-1]:.4f}")
print(f"pocket w: {ex2_run.pocket_weights}, b: {ex2_run.pocket_bias:.4f}, "
      f"accuracy: {ex2_run.pocket_accuracy:.4f} (first reached in epoch {ex2_run.pocket_epoch})")
print(f"epochs run: {ex2_run.epochs_run}")
print(f"updates in the first and last epoch: {ex2_run.update_history[0]}, "
      f"{ex2_run.update_history[-1]}; range over the run: {min(ex2_run.update_history)} "
      f"to {max(ex2_run.update_history)}")

# %% [markdown]
# ### C — Figures

# %%
ex2_misclassified = predict(ex2_points, ex2_run.pocket_weights, ex2_run.pocket_bias) != ex2_labels

plot_decision_figure(
    ex2_points, ex2_labels,
    [(ex2_run.weights, ex2_run.bias, "black", "Final boundary"),
     (ex2_run.pocket_weights, ex2_run.pocket_bias, "#55A868", "Pocket boundary")],
    ex2_misclassified,
    "Figure 5 — Exercise 2: final and pocket boundaries\n(rings mark the points the pocket misses)",
    "fig5.png",
)

plot_accuracy_figure(
    [(ex2_run.accuracy_history, "#4C72B0", "Current weights"),
     (ex2_run.pocket_accuracy_history, "#55A868", "Pocket (best so far)")],
    "Figure 6 — Exercise 2: current and pocket accuracy per epoch", "fig6.png",
    legend_loc="center right",
)

# %% [markdown]
# ### D — Analysis

# %%
# Where does the final boundary sit? Distance from a point to w · x + b = 0 is
# |w · point + b| / ||w||, so compare that against the size of the cloud itself.
cloud_centre = np.array([3.5, 3.5])
final_distance = abs(ex2_run.weights @ cloud_centre + ex2_run.bias) / np.linalg.norm(ex2_run.weights)
pocket_distance = (abs(ex2_run.pocket_weights @ cloud_centre + ex2_run.pocket_bias)
                   / np.linalg.norm(ex2_run.pocket_weights))
cloud_radius = np.linalg.norm(ex2_points - cloud_centre, axis=1).max()

print(f"mean ||x||: {np.linalg.norm(ex2_points, axis=1).mean():.3f}")
print(f"|b| / ||w|| (final): {abs(ex2_run.bias) / np.linalg.norm(ex2_run.weights):.3f}")
print(f"distance from the cloud centre to the final boundary:  {final_distance:.3f}")
print(f"distance from the cloud centre to the pocket boundary: {pocket_distance:.3f}")
print(f"furthest point from the cloud centre: {cloud_radius:.3f}")
print(f"share of points the final weights call class 1: "
      f"{predict(ex2_points, ex2_run.weights, ex2_run.bias).mean():.4f}")
print(f"accuracy range over the epochs: {min(ex2_run.accuracy_history):.4f} "
      f"to {max(ex2_run.accuracy_history):.4f}")
