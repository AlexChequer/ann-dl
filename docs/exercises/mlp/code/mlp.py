# %% [markdown]
# # MLP — from scratch, binary and multi-class
#
# Exercise 1 is solved by hand in the report. This notebook covers Exercises 2–4.

# %%
from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# One generator for the whole report. Every random draw below (all weight
# initialisations) comes from it, in the order the cells run.
rng = np.random.default_rng(42)

# Resolve paths whether this runs as a script (__file__ exists) or in a
# Jupyter kernel (it does not) — walk up to the repo root, marked by mkdocs.yml.
try:
    code_dir = Path(__file__).resolve().parent
except NameError:
    repo_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "mkdocs.yml").exists())
    code_dir = repo_root / "docs" / "exercises" / "mlp" / "code"

figures_dir = code_dir.parent / "figures"
figures_dir.mkdir(exist_ok=True)

SPLIT_SEED = 42
CLASS_COLORS = ["#4C72B0", "#C44E52", "#55A868"]


def save(fig, filename):
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def standardise(train, test):
    """Scale both sets with the mean and std of the training set only."""
    mean, std = train.mean(axis=0), train.std(axis=0)
    return (train - mean) / std, (test - mean) / std


# %% [markdown]
# ## The MLP
#
# Shared by Exercises 2, 3 and 4 without modification. Only the output
# activation, the loss and the hyper-parameters change between them.
#
# Conventions: a batch `X` has one sample per row; layer `l` holds
# `W` of shape `(n_in, n_out)` and `b` of shape `(n_out,)`, so `Z = A @ W + b`.
# Hidden layers use `tanh`.

# %%
class Layer(NamedTuple):
    W: np.ndarray
    b: np.ndarray


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def softmax(u):
    shifted = np.exp(u - u.max(axis=1, keepdims=True))  # subtract max: no overflow
    return shifted / shifted.sum(axis=1, keepdims=True)


EPS = 1e-12  # keeps log() finite when a probability reaches 0 or 1


def binary_cross_entropy(y_hat, Y):
    return -np.mean(Y * np.log(y_hat + EPS) + (1 - Y) * np.log(1 - y_hat + EPS))


def categorical_cross_entropy(y_hat, Y):
    return -np.mean(np.sum(Y * np.log(y_hat + EPS), axis=1))


def init_layers(layer_sizes):
    """Xavier-normal weights, zero biases. layer_sizes = [n_in, hidden..., n_out]."""
    return [
        Layer(rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_in, n_out)), np.zeros(n_out))
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def forward(layers, X, output_activation):
    """Return the activations of every layer, input first, prediction last."""
    activations = [X]
    for W, b in layers[:-1]:
        activations.append(np.tanh(activations[-1] @ W + b))
    W, b = layers[-1]
    activations.append(output_activation(activations[-1] @ W + b))
    return activations


def backward(layers, activations, Y):
    """Gradients of the mean loss for every layer, by the chain rule.

    For sigmoid + binary cross-entropy and for softmax + categorical
    cross-entropy the output gradient is the same, dL/du = (y_hat - y) / N
    (derived in the report), so one backward pass serves both.
    """
    n_samples = Y.shape[0]
    delta = (activations[-1] - Y) / n_samples  # dL/du at the output layer
    grads = []
    for index in reversed(range(len(layers))):
        a_prev = activations[index]
        grads.append(Layer(a_prev.T @ delta, delta.sum(axis=0)))
        if index > 0:
            # through W into the previous tanh layer: tanh'(z) = 1 - tanh(z)^2
            delta = (delta @ layers[index].W.T) * (1.0 - a_prev**2)
    return grads[::-1]


def update(layers, grads, learning_rate):
    """One gradient-descent step; returns new layers, the old ones are untouched."""
    return [
        Layer(layer.W - learning_rate * grad.W, layer.b - learning_rate * grad.b)
        for layer, grad in zip(layers, grads)
    ]


class TrainedMLP(NamedTuple):
    layers: list
    loss_history: list


def train(layer_sizes, X, Y, output_activation: Callable, loss: Callable,
          learning_rate, epochs):
    """Full-batch gradient descent; the loss is recorded once per epoch."""
    layers = init_layers(layer_sizes)
    loss_history = []
    for _ in range(epochs):
        activations = forward(layers, X, output_activation)
        loss_history.append(float(loss(activations[-1], Y)))
        layers = update(layers, backward(layers, activations, Y), learning_rate)
    return TrainedMLP(layers, loss_history)


def predict_proba(layers, X, output_activation):
    return forward(layers, X, output_activation)[-1]


def count_parameters(layer_sizes):
    return sum((n_in + 1) * n_out for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]))


def plot_loss(ax, history, label):
    ax.plot(np.arange(1, len(history) + 1), history, linewidth=2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (cross-entropy)")
    ax.grid(alpha=0.3)


# %% [markdown]
# ## Exercise 2 — binary classification a line cannot do
#
# ### A — Data

# %%
ex2_points, ex2_clusters = make_blobs(
    n_samples=[500, 250, 250], centers=[[0, 0], [3, 3], [-3, -3]],
    cluster_std=1.2, random_state=42,
)
ex2_labels = (ex2_clusters > 0).astype(int)  # class 1 = both outer clusters
ex2_X_train_raw, ex2_X_test_raw, ex2_y_train, ex2_y_test = train_test_split(
    ex2_points, ex2_labels, test_size=0.2, random_state=SPLIT_SEED, stratify=ex2_labels,
)
ex2_X_train, ex2_X_test = standardise(ex2_X_train_raw, ex2_X_test_raw)
print("class counts:", np.bincount(ex2_labels), " train/test:", len(ex2_y_train), len(ex2_y_test))


def scatter_classes(ax, points, labels, **kwargs):
    for class_index in np.unique(labels):
        in_class = points[labels == class_index]
        ax.scatter(in_class[:, 0], in_class[:, 1], color=CLASS_COLORS[class_index],
                   label=f"Class {class_index}", **kwargs)


fig, ax = plt.subplots(figsize=(6, 6))
scatter_classes(ax, ex2_points, ex2_labels, s=12, alpha=0.6)
ax.set_title("Figure 1 — Exercise 2 data (1000 points)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.legend()
save(fig, "fig1.png")

# %% [markdown]
# ### B — Linear baseline

# %%
ex2_baseline = LogisticRegression().fit(ex2_X_train, ex2_y_train)
ex2_baseline_acc = ex2_baseline.score(ex2_X_test, ex2_y_test)
print(f"logistic regression test accuracy: {ex2_baseline_acc:.4f}")
print(f"share of class 0 in the test set:  {np.mean(ex2_y_test == 0):.4f}")
print(f"baseline coefficients: {ex2_baseline.coef_.ravel()}, intercept {ex2_baseline.intercept_}")

# %% [markdown]
# ### C — MLP from scratch

# %%
EX2_SIZES = [2, 8, 1]
EX2_LEARNING_RATE = 0.5
EX2_EPOCHS = 2000

ex2_mlp = train(EX2_SIZES, ex2_X_train, ex2_y_train.reshape(-1, 1),
                sigmoid, binary_cross_entropy, EX2_LEARNING_RATE, EX2_EPOCHS)
ex2_pred = (predict_proba(ex2_mlp.layers, ex2_X_test, sigmoid).ravel() >= 0.5).astype(int)
ex2_mlp_acc = np.mean(ex2_pred == ex2_y_test)
print(f"architecture {EX2_SIZES}, {count_parameters(EX2_SIZES)} parameters")
print(f"loss: epoch 1 {ex2_mlp.loss_history[0]:.4f}, final {ex2_mlp.loss_history[-1]:.4f}")
print(f"MLP test accuracy: {ex2_mlp_acc:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
plot_loss(ax, ex2_mlp.loss_history, f"MLP {EX2_SIZES}")
ax.set_title("Figure 2 — Exercise 2 training loss")
ax.legend()
save(fig, "fig2.png")

# %%
# Decision boundary: evaluate the network on a grid in the original units.
train_mean, train_std = ex2_X_train_raw.mean(axis=0), ex2_X_train_raw.std(axis=0)
grid_x1, grid_x2 = np.meshgrid(np.linspace(-8, 8, 400), np.linspace(-8, 8, 400))
grid = np.column_stack([grid_x1.ravel(), grid_x2.ravel()])
grid_proba = predict_proba(ex2_mlp.layers, (grid - train_mean) / train_std, sigmoid)
grid_class = (grid_proba.reshape(grid_x1.shape) >= 0.5).astype(int)

fig, ax = plt.subplots(figsize=(6, 6))
ax.contourf(grid_x1, grid_x2, grid_class, levels=[-0.5, 0.5, 1.5],
            colors=CLASS_COLORS[:2], alpha=0.18)
ax.contour(grid_x1, grid_x2, grid_class, levels=[0.5], colors="black", linewidths=1.5)
scatter_classes(ax, ex2_X_test_raw, ex2_y_test, s=18, edgecolor="white", linewidth=0.5)
ax.set_title("Figure 3 — MLP decision regions, test points")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.legend(title="Test points")
save(fig, "fig3.png")

# %% [markdown]
# ## Exercise 3 — multi-class, same network
#
# ### A — Data

# %%
ex3_points, ex3_labels = make_classification(
    n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
    n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=42,
)
ex3_X_train, ex3_X_test, ex3_y_train, ex3_y_test = train_test_split(
    ex3_points, ex3_labels, test_size=0.2, random_state=SPLIT_SEED, stratify=ex3_labels,
)
ex3_X_train, ex3_X_test = standardise(ex3_X_train, ex3_X_test)
ex3_Y_train = np.eye(3)[ex3_y_train]  # one-hot targets for softmax
print("class counts:", np.bincount(ex3_labels), " train/test:", len(ex3_y_train), len(ex3_y_test))

# %% [markdown]
# ### C — Training (B, the adaptation, is only the call below: 3 outputs, softmax, CCE)

# %%
EX3_SIZES = [4, 16, 3]
LEARNING_RATE = 0.5
EPOCHS = 3000


def train_ex3_data(layer_sizes):
    """Train on the Exercise 3 data with the fixed budget; return (model, test accuracy)."""
    model = train(layer_sizes, ex3_X_train, ex3_Y_train, softmax,
                  categorical_cross_entropy, LEARNING_RATE, EPOCHS)
    predictions = predict_proba(model.layers, ex3_X_test, softmax).argmax(axis=1)
    return model, np.mean(predictions == ex3_y_test)


ex3_mlp, ex3_mlp_acc = train_ex3_data(EX3_SIZES)
ex3_baseline_acc = LogisticRegression().fit(ex3_X_train, ex3_y_train).score(ex3_X_test, ex3_y_test)
print(f"architecture {EX3_SIZES}, {count_parameters(EX3_SIZES)} parameters")
print(f"final training loss: {ex3_mlp.loss_history[-1]:.4f}")
print(f"MLP test accuracy: {ex3_mlp_acc:.4f}")
print(f"logistic regression test accuracy: {ex3_baseline_acc:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
plot_loss(ax, ex3_mlp.loss_history, f"MLP {EX3_SIZES}")
ax.set_title("Figure 4 — Exercise 3 training loss")
ax.legend()
save(fig, "fig4.png")

# %%
ex3_pred = predict_proba(ex3_mlp.layers, ex3_X_test, softmax).argmax(axis=1)
ex3_confusion = confusion_matrix(ex3_y_test, ex3_pred)
print(ex3_confusion)

fig, ax = plt.subplots(figsize=(5, 4.5))
image = ax.imshow(ex3_confusion, cmap="Blues")
for (row, col), count in np.ndenumerate(ex3_confusion):
    is_dark = count > ex3_confusion.max() / 2
    ax.text(col, row, count, ha="center", va="center", color="white" if is_dark else "black")
ax.set_xticks(range(3), [f"Class {k}" for k in range(3)])
ax.set_yticks(range(3), [f"Class {k}" for k in range(3)])
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Figure 5 — Ex. 3 confusion matrix (test)")
fig.colorbar(image, ax=ax, label="Test samples")
save(fig, "fig5.png")

# %% [markdown]
# ### D — Supporting numbers for the analysis

# %%
# The baseline's mistakes, for comparison with Figure 5.
ex3_baseline = LogisticRegression().fit(ex3_X_train, ex3_y_train)
print("logistic regression confusion matrix:")
print(confusion_matrix(ex3_y_test, ex3_baseline.predict(ex3_X_test)))

# How much each pair of classes overlaps in 4D: for every training point, the
# class of its nearest other training point. A cross-class nearest neighbour
# means the two classes share that neighbourhood.
squared_distances = ((ex3_X_train[:, None, :] - ex3_X_train[None, :, :]) ** 2).sum(axis=2)
np.fill_diagonal(squared_distances, np.inf)
neighbour_class = ex3_y_train[squared_distances.argmin(axis=1)]
for first, second in ((0, 1), (0, 2), (1, 2)):
    shared = np.sum((ex3_y_train == first) & (neighbour_class == second)) + \
             np.sum((ex3_y_train == second) & (neighbour_class == first))
    print(f"classes {first} and {second}: {shared} points whose nearest neighbour is the other class")

# %% [markdown]
# ## Exercise 4 — does a second hidden layer help?
#
# ### A — Training: one change, a second hidden layer of the same width

# %%
EX4_SIZES = [4, 16, 16, 3]
ex4_mlp, ex4_mlp_acc = train_ex3_data(EX4_SIZES)
print(f"architecture {EX4_SIZES}, {count_parameters(EX4_SIZES)} parameters")
print(f"final training loss: {ex4_mlp.loss_history[-1]:.4f}")
print(f"MLP test accuracy: {ex4_mlp_acc:.4f}")

# %% [markdown]
# ### B — Comparison over three initialisations
#
# The data and split are fixed, so "seed" means the weight initialisation. Each
# run is a fresh draw from the shared `rng`; run 1 is the model trained above.

# %%
N_RUNS = 3
runs = {"Ex. 3": [(ex3_mlp, ex3_mlp_acc)], "Ex. 4": [(ex4_mlp, ex4_mlp_acc)]}
for _ in range(N_RUNS - 1):
    runs["Ex. 3"].append(train_ex3_data(EX3_SIZES))
    runs["Ex. 4"].append(train_ex3_data(EX4_SIZES))

for name, sizes in (("Ex. 3", EX3_SIZES), ("Ex. 4", EX4_SIZES)):
    final_losses = [model.loss_history[-1] for model, _ in runs[name]]
    accuracies = [acc for _, acc in runs[name]]
    train_accuracies = [
        np.mean(predict_proba(model.layers, ex3_X_train, softmax).argmax(axis=1) == ex3_y_train)
        for model, _ in runs[name]
    ]
    print(f"{name} train acc {np.round(train_accuracies, 4)} mean {np.mean(train_accuracies):.4f}, "
          f"test acc std {np.std(accuracies) * 100:.2f} points")
    print(f"{name} {sizes}: {len(sizes) - 2} hidden, {count_parameters(sizes)} params | "
          f"final loss {np.round(final_losses, 4)} mean {np.mean(final_losses):.4f} | "
          f"test acc {np.round(accuracies, 4)} mean {np.mean(accuracies):.4f}")
mean_difference = (np.mean([acc for _, acc in runs["Ex. 4"]])
                   - np.mean([acc for _, acc in runs["Ex. 3"]]))
print(f"mean accuracy difference (Ex. 4 - Ex. 3): {mean_difference * 100:+.2f} points")

fig, ax = plt.subplots(figsize=(7, 4))
for name, sizes, color in (("Ex. 3", EX3_SIZES, CLASS_COLORS[0]),
                           ("Ex. 4", EX4_SIZES, CLASS_COLORS[1])):
    for run_index, (model, _) in enumerate(runs[name]):
        ax.plot(np.arange(1, EPOCHS + 1), model.loss_history, color=color, linewidth=1.5,
                alpha=1.0 if run_index == 0 else 0.35,
                label=f"{name} {sizes}" if run_index == 0 else None)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (cross-entropy)")
ax.set_title("Figure 6 — One vs. two hidden layers (3 initialisations each)")
ax.grid(alpha=0.3)
ax.legend()
save(fig, "fig6.png")
