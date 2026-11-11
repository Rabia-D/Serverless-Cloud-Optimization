
import os
import pandas as pd

# ==========================================
# SERVERLESS SIMULATOR — VERSION 1
# ==========================================

DATA_PATH = "models/policy_comparison_details.csv"
MAX_WARM_APPS = 4
COLD_START_PENALTY_MS = 500

df = pd.read_csv(DATA_PATH)
df = df.sort_values(["window", "app"]).reset_index(drop=True)

required = {
    "app", "window", "target",
    "static_warm", "recent_warm", "ml_warm"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Each policy must operate on the same workload.
POLICIES = {
    "Static": "static_warm",
    "Recent activity": "recent_warm",
    "ML prediction": "ml_warm"
}


def simulate_policy(data, policy_name, decision_column):
    warm_apps = set()
    records = []

    total_cold_start_events = 0
    total_active_windows = 0
    total_penalty_ms = 0

    for window, group in data.groupby("window", sort=True):

        selected = set(
            group.loc[
                group[decision_column] == 1, "app"
            ]
        )

        if len(selected) > MAX_WARM_APPS:
            raise ValueError(
                f"{policy_name}: capacity exceeded "
                f"at window {window}"
            )

        # Decisions made at t apply to the next
        # window, whose demand is stored in target.
        unloaded = warm_apps - selected
        loaded = selected - warm_apps

        warm_apps = selected.copy()

        for _, row in group.iterrows():
            app = row["app"]
            invocations = int(row["target"])

            is_warm = app in warm_apps
            is_active = invocations > 0

            # Window-level approximation:
            # one cold-start event for an active
            # application that is not prewarmed.
            cold_start = int(is_active and not is_warm)

            total_cold_start_events += cold_start
            total_active_windows += int(is_active)

            penalty = (
                cold_start * COLD_START_PENALTY_MS
            )
            total_penalty_ms += penalty

            records.append({
                "policy": policy_name,
                "window": window,
                "app": app,
                "invocations": invocations,
                "warm": int(is_warm),
                "cold_start_event": cold_start,
                "cold_start_penalty_ms": penalty,
                "loaded_at_boundary": int(app in loaded),
                "unloaded_at_boundary": int(app in unloaded)
            })

    cold_start_rate = (
        total_cold_start_events / total_active_windows
        if total_active_windows else 0
    )

    details = pd.DataFrame(records)

    summary = {
        "policy": policy_name,
        "active_app_windows": total_active_windows,
        "simulated_cold_start_events": total_cold_start_events,
        "simulated_cold_start_rate": cold_start_rate,
        "avg_warm_apps": (
            details.groupby("window")["warm"].sum().mean()
        ),
        "total_penalty_ms": total_penalty_ms,
        "load_operations": int(
            details["loaded_at_boundary"].sum()
        ),
        "unload_operations": int(
            details["unloaded_at_boundary"].sum()
        )
    }

    return summary, details


# ==========================================
# RUN ALL POLICIES
# ==========================================

summaries = []
all_details = []

for policy_name, column in POLICIES.items():
    summary, details = simulate_policy(
        df, policy_name, column
    )

    summaries.append(summary)
    all_details.append(details)

summary_df = pd.DataFrame(summaries)
details_df = pd.concat(all_details, ignore_index=True)

os.makedirs("models", exist_ok=True)

summary_df.to_csv(
    "models/simulator_summary.csv",
    index=False
)

details_df.to_csv(
    "models/simulator_details.csv",
    index=False
)

print("\n========== SIMULATOR RESULTS ==========")
print(summary_df.to_string(index=False))
print("=======================================")

print("\nSaved:")
print("- models/simulator_summary.csv")
print("- models/simulator_details.csv")
print("\nSimulation complete!")


