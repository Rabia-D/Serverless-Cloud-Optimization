
import os
import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

pred = pd.read_csv(
    "models/gradient_boosting_predictions.csv"
)

raw = pd.read_csv(
    "data/azure_2021_sample_complete.csv"
)

pred = pred.sort_values(
    ["window", "app"]
).reset_index(drop=True)

raw = raw.sort_values(["window", "app"])

MAX_WARM = 4
ML_THRESHOLD = 10

# ==========================================
# 2. STATIC BASELINE
# ==========================================

# Use only historical training data.
first_test_window = pred["window"].min()

train = raw[
    raw["window"] < first_test_window
]

top_apps = (
    train.groupby("app")["invocations"]
    .sum()
    .nlargest(MAX_WARM)
    .index
)

pred["static_warm"] = (
    pred["app"].isin(top_apps).astype(int)
)

# ==========================================
# 3. RECENT-ACTIVITY BASELINE
# ==========================================

# At decision window t, use activity from t
# to decide which apps to keep warm for t+1.
# No future information is used.

recent = raw[
    ["app", "window", "invocations"]
].copy()

recent = recent.rename(
    columns={"invocations": "recent_count"}
)

pred = pred.merge(
    recent,
    on=["app", "window"],
    how="left",
    validate="one_to_one"
)

if pred["recent_count"].isna().any():
    raise ValueError(
        "Missing recent activity for some windows."
    )

pred["recent_warm"] = 0

for window, group in pred.groupby("window"):
    selected = group.nlargest(
        MAX_WARM,
        "recent_count"
    )

    pred.loc[
        selected.index, "recent_warm"
    ] = 1

# ==========================================
# 4. ML PREDICTION POLICY
# ==========================================

pred["ml_warm"] = 0

for window, group in pred.groupby("window"):

    eligible = group[
        group["predicted"] >= ML_THRESHOLD
    ]

    selected = eligible.nlargest(
        MAX_WARM,
        "predicted"
    )

    pred.loc[
        selected.index, "ml_warm"
    ] = 1

# ==========================================
# 5. EVALUATE ALL THREE POLICIES
# ==========================================

active = pred["target"] > 0
active_count = int(active.sum())

results = []

for name, column in [
    ("Static", "static_warm"),
    ("Recent activity", "recent_warm"),
    ("ML prediction", "ml_warm"),
]:

    missed = active & (pred[column] == 0)

    missed_windows = int(missed.sum())

    missed_rate = (
        missed_windows / active_count
        if active_count > 0 else 0
    )

    warm_occupancy = pred[column].mean()

    average_warm_apps = (
        pred.groupby("window")[column]
        .sum()
        .mean()
    )

    results.append({
        "policy": name,
        "active_windows": active_count,
        "missed_active_windows": missed_windows,
        "cold_start_window_rate": missed_rate,
        "warm_slot_occupancy": warm_occupancy,
        "average_warm_apps": average_warm_apps,
    })

summary = pd.DataFrame(results)

# ==========================================
# 6. PRINT RESULTS
# ==========================================

print("\n========== POLICY COMPARISON ==========")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("=======================================")

# ==========================================
# 7. SAVE RESULTS
# ==========================================

os.makedirs("models", exist_ok=True)

pred.to_csv(
    "models/policy_comparison_details.csv",
    index=False
)

summary.to_csv(
    "models/policy_comparison_summary.csv",
    index=False
)

print("\nSaved:")
print("- models/policy_comparison_details.csv")
print("- models/policy_comparison_summary.csv")
print("\nComparison complete!")

