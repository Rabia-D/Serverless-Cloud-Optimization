
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. LOAD AZURE FUNCTIONS 2021 DATASET
# ==========================================

df = pd.read_csv("data/azure_2021_sample_complete.csv")
df = df.sort_values(["app", "window"]).reset_index(drop=True)

print("Dataset loaded:", df.shape)
print("Applications:", df["app"].nunique())

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# Historical invocation counts for each application
grouped = df.groupby("app")["invocations"]

for lag in [0, 1, 2, 3, 6, 12]:
    df[f"lag_{lag}"] = grouped.shift(lag)

# Rolling average invocation activity
for window_size in [3, 6, 12]:
    df[f"rolling_mean_{window_size}"] = (
        df.groupby("app")["invocations"]
        .transform(
            lambda s: s.rolling(
                window_size, min_periods=1
            ).mean()
        )
    )

# Idle duration: consecutive previous windows with no invocations
def calculate_idle_duration(series):
    idle = []
    count = 0

    for value in series:
        if value == 0:
            count += 1
        else:
            count = 0

        idle.append(count)

    return pd.Series(idle, index=series.index)

df["idle_duration"] = (
    df.groupby("app")["invocations"]
    .transform(calculate_idle_duration)
)

# Target: invocation count in the NEXT 5-minute window
df["target"] = grouped.shift(-1)

# Remove rows without sufficient history or future target
df = df.dropna().copy()

features = [
    col for col in df.columns
    if col.startswith("lag_")
    or col.startswith("rolling_mean_")
]

features.append("idle_duration")

print("Features:", features)

# ==========================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================

# All applications use the same time boundary
unique_windows = np.sort(df["window"].unique())
split_window = unique_windows[int(len(unique_windows) * 0.8)]

train = df[df["window"] < split_window]
test = df[df["window"] >= split_window]

X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]

print("Training rows:", len(train))
print("Testing rows:", len(test))

# ==========================================
# 4. TRAIN GRADIENT BOOSTING MODEL
# ==========================================

model = HistGradientBoostingRegressor(
    max_iter=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

print("\nTraining model...")
model.fit(X_train, y_train)

# ==========================================
# 5. EVALUATE MODEL
# ==========================================

predictions = np.maximum(model.predict(X_test), 0)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Simple baseline: next window equals current window
baseline_predictions = X_test["lag_0"].to_numpy()

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\n========== MODEL RESULTS ==========")
print(f"Gradient Boosting MAE: {mae:.4f}")
print(f"Gradient Boosting RMSE: {rmse:.4f}")
print(f"Naive Baseline MAE: {baseline_mae:.4f}")
print(f"Naive Baseline RMSE: {baseline_rmse:.4f}")
print("===================================")

# ==========================================
# 6. SAVE MODEL AND PREDICTIONS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

joblib.dump(
    model,
    "models/gradient_boosting_baseline.joblib"
)

results = test[["app", "window", "target"]].copy()
results["predicted"] = predictions

results.to_csv(
    "models/gradient_boosting_predictions.csv",
    index=False
)

# ==========================================
# 7. VISUALIZE ACTUAL VS PREDICTED
# ==========================================

# Choose the application with the highest test activity
app = (
    results.groupby("app")["target"]
    .sum()
    .idxmax()
)

subset = (
    results[results["app"] == app]
    .sort_values("window")
    .head(200)
)

plt.figure(figsize=(12, 5))

plt.plot(
    subset["window"],
    subset["target"],
    label="Actual Invocations",
    linewidth=1.5
)

plt.plot(
    subset["window"],
    subset["predicted"],
    label="Predicted Invocations",
    linewidth=1.5
)

plt.xlabel("Five-minute Window")
plt.ylabel("Invocation Count")
plt.title(
    "Azure Functions 2021: Actual vs Predicted Invocations"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures/gradient_boosting_predictions.png",
    dpi=200
)

plt.close()

print("\nSaved:")
print("- models/gradient_boosting_baseline.joblib")
print("- models/gradient_boosting_predictions.csv")
print("- figures/gradient_boosting_predictions.png")

print("\nTraining complete!")

