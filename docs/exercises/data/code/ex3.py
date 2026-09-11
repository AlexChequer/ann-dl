# %% [markdown]
# # Exercise 3 — Preparing real-world data for a tanh network

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

RANDOM_STATE = 42

try:
    code_dir = Path(__file__).resolve().parent
except NameError:
    repo_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "mkdocs.yml").exists())
    code_dir = repo_root / "docs" / "exercises" / "data" / "code"

figures_dir = code_dir.parent / "figures"
figures_dir.mkdir(exist_ok=True)

SPENDING_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERICAL_COLUMNS = ["Age", *SPENDING_COLUMNS]
CATEGORICAL_COLUMNS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
IDENTIFIER_COLUMNS = ["PassengerId", "Cabin", "Name"]
TARGET_COLUMN = "Transported"

raw = pd.read_csv(code_dir.parent / "data" / "train.csv")

# %% [markdown]
# ## A — Get to know the data

# %%
target_balance = raw[TARGET_COLUMN].value_counts(normalize=True)
print(f"Rows: {len(raw)}   Columns: {raw.shape[1]}\n")
print("Class balance of Transported:")
for value, share in target_balance.items():
    print(f"  {value}:  {share:.2%}  ({raw[TARGET_COLUMN].value_counts()[value]} rows)")

print("\nMissing values per column:")
missing = pd.DataFrame({
    "missing": raw.isna().sum(),
    "percent": (raw.isna().sum() / len(raw) * 100).round(2),
}).sort_values("missing", ascending=False)
print(missing[missing["missing"] > 0].to_string())

print("\nSpending columns — mean, median, maximum (whole file):")
spending_stats = raw[SPENDING_COLUMNS].agg(["mean", "median", "max"]).T.round(2)
print(spending_stats.to_string())

# %% [markdown]
# ## B — Split before you transform

# %%
features = raw.drop(columns=[*IDENTIFIER_COLUMNS, TARGET_COLUMN])
target = raw[TARGET_COLUMN]

train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE
)

print(f"Train: {train_features.shape}   Test: {test_features.shape}")
print(f"Positive share — train {train_target.mean():.4f}, test {test_target.mean():.4f}")

food_court_mean = train_features["FoodCourt"].mean()
food_court_median = train_features["FoodCourt"].median()
print(f"\nFoodCourt on the training set, before transforming:"
      f"  mean {food_court_mean:.2f}   median {food_court_median:.2f}")

# %% [markdown]
# ## C — Preprocess

# %%
# Imputation. Median for numerical columns because the spending distributions are
# heavily right-skewed and the mean would be dragged up by the extreme spenders;
# most-frequent for categorical columns, which is the only sensible constant for
# an unordered label. Both are fitted on the training set only.
numerical_imputer = SimpleImputer(strategy="median").fit(train_features[NUMERICAL_COLUMNS])
categorical_imputer = SimpleImputer(strategy="most_frequent").fit(
    train_features[CATEGORICAL_COLUMNS].astype(object)
)


def impute(frame):
    numerical = pd.DataFrame(
        numerical_imputer.transform(frame[NUMERICAL_COLUMNS]),
        columns=NUMERICAL_COLUMNS, index=frame.index,
    )
    categorical = pd.DataFrame(
        categorical_imputer.transform(frame[CATEGORICAL_COLUMNS].astype(object)),
        columns=CATEGORICAL_COLUMNS, index=frame.index,
    ).astype(str)
    return numerical, categorical


train_numerical, train_categorical = impute(train_features)
test_numerical, test_categorical = impute(test_features)

# Feature engineering: total spend across the five services, from the raw amounts.
for frame in (train_numerical, test_numerical):
    frame["TotalSpend"] = frame[SPENDING_COLUMNS].sum(axis=1)

raw_food_court_train = train_numerical["FoodCourt"].copy()

# Heavy tails: log(1 + x) compresses the long right tail of the money columns.
MONEY_COLUMNS = [*SPENDING_COLUMNS, "TotalSpend"]
for frame in (train_numerical, test_numerical):
    frame[MONEY_COLUMNS] = np.log1p(frame[MONEY_COLUMNS])

# Categorical encoding. handle_unknown="ignore" means a category seen only in the
# test set becomes all-zeros across that feature's columns instead of raising.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train_categorical)
train_encoded = encoder.transform(train_categorical)
test_encoded = encoder.transform(test_categorical)

# Scaling to [-1, 1], matching the output range of tanh. Fitted on train only.
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_numerical)
train_scaled = scaler.transform(train_numerical)
test_scaled = scaler.transform(test_numerical)

train_matrix = np.hstack([train_scaled, train_encoded])
test_matrix = np.hstack([test_scaled, test_encoded])

feature_names = [*train_numerical.columns, *encoder.get_feature_names_out(CATEGORICAL_COLUMNS)]

print(f"Encoded categories: {list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))}")
print(f"\nScaled numerical range — train [{train_scaled.min():.3f}, {train_scaled.max():.3f}]")
print(f"Scaled numerical range — test  [{test_scaled.min():.3f}, {test_scaled.max():.3f}]")
print(f"Full matrix range      — train [{train_matrix.min():.3f}, {train_matrix.max():.3f}]")
print(f"Full matrix range      — test  [{test_matrix.min():.3f}, {test_matrix.max():.3f}]")

# %% [markdown]
# ## D — Verify and visualize

# %%
def plot_figure_6():
    """Figure 6 — FoodCourt before and after log(1+x) plus scaling to [-1, 1]."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(raw_food_court_train, bins=60, color="#C44E52")
    axes[0].set_title("Before — raw FoodCourt spend")
    axes[0].set_xlabel("credits spent")

    processed = train_matrix[:, feature_names.index("FoodCourt")]
    axes[1].hist(processed, bins=60, color="#4C72B0")
    axes[1].set_title("After — log(1+x), then scaled to [-1, 1]")
    axes[1].set_xlabel("scaled value")

    for ax in axes:
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)

    fig.suptitle("Figure 6 — a heavy-tailed feature before and after preprocessing")
    fig.savefig(figures_dir / "fig6.png", dpi=150, bbox_inches="tight")
    return fig


plot_figure_6()

print(f"NaN remaining — train {np.isnan(train_matrix).sum()}, test {np.isnan(test_matrix).sum()}")
print(f"Final training feature matrix shape: {train_matrix.shape}")
print(f"Final test feature matrix shape:     {test_matrix.shape}")
print(f"Features ({len(feature_names)}): {feature_names}")
