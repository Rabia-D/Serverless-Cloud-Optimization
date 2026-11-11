
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. LOAD ML PREDICTIONS
# ==========================================

df = pd.read_csv(
    "models/gradient_boosting_predictions.csv"
)

df = df.sort_values(["window", "app"]).reset_index(drop=True)

print("Predictions loaded:", len(df))
print("Applications:", df["app"].nunique())

# ==========================================
# 2. DEFINE RESOURCE POLICY
# ==========================================

# Initial experimental assumptions
MAX_WARM_APPS = 4
PREDICTION_THRESHOLD = 10

# Keep up to four application environments warm.
# Prioritize applications with predicted demand >= 10.
# These values are illustrative, not Azure measurements.

df["keep_warm"] = 0

for window, group in df.groupby("window"):
    eligible = group[
        group["predicted"] >= PREDICTION_THRESHOLD
    ]

    selected = eligible.nlargest(
        MAX_WARM_APPS, "predicted"
    )

    df.loc[selected.index, "keep_warm"] = 1

# ==========================================
# 3. CALCULATE INITIAL PROXY METRICS
# ==========================================

# An inactive application that becomes active while
# unloaded is counted as one simulated cold-start event.
# This is a window-level approximation, not an actual
# measurement of Azure cold starts.

df["simulated_cold_start"] = (
    (df["target"] > 0) &
    (df["keep_warm"] == 0)
).astype(int)

df["warm_slots_used"] = df["keep_warm"]

cold_starts = int(df["simulated_cold_start"].sum())
active_windows = int((df["target"] > 0).sum())
warm_slots = int(df["warm_slots_used"].sum())
total_slots = len(df)

cold_start_rate = (
    cold_starts / active_windows
    if active_windows else 0
)

warm_utilization = (
    warm_slots / total_slots
    if total_slots else 0
)

# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

print("\n========== ML RESOURCE POLICY ==========")
print("Maximum warm applications:", MAX_WARM_APPS)
print("Prediction threshold:", PREDICTION_THRESHOLD)
print("Active application-windows:", active_windows)
print("Simulated cold-start windows:", cold_starts)
print(f"Cold-start window rate: {cold_start_rate:.2%}")
print(f"Average warm-slot occupancy: {warm_utilization:.2%}")
print("========================================")

# ==========================================
# 5. SAVE RESULTS
# ==========================================

os.makedirs("models", exist_ok=True)

df.to_csv(
    "models/ml_resource_policy_results.csv",
    index=False
)

print("\nSaved: models/ml_resource_policy_results.csv")

