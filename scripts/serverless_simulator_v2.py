
import os
import pandas as pd

# ==========================================
# SERVERLESS SIMULATOR — VERSION 2
# ==========================================

DATA_PATH = "models/policy_comparison_details.csv"
MAX_WARM_APPS = 4
COLD_START_PENALTY_MS = 500

POLICIES = {
    "Static": "static_warm",
    "Recent activity": "recent_warm",
    "ML prediction": "ml_warm",
}

df = pd.read_csv(DATA_PATH)
df = df.sort_values(["window", "app"]).reset_index(drop=True)

required = {
    "app", "window", "target",
    "static_warm", "recent_warm", "ml_warm"
}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required - set(df.columns)}"
    )

if df.duplicated(["app", "window"]).any():
    raise ValueError("Duplicate application-window records.")

def simulate(data, policy_name, policy_column):
    # Persistent container state across windows.
    warm_apps = set()
    records = []

    for window, group in data.groupby("window", sort=True):

        # Desired prewarm selection for the next
        # invocation window.
        selected = set(
            group.loc[
                group[policy_column] == 1, "app"
            ]
        )

        if len(selected) > MAX_WARM_APPS:
            raise ValueError("Warm capacity exceeded.")

        # Transition from previous warm state.
        evicted = warm_apps - selected
        prewarmed = selected - warm_apps

        # Apply policy decisions before demand arrives.
        warm_apps.difference_update(evicted)
        warm_apps.update(prewarmed)

        # Demand for the following window.
        for _, row in group.iterrows():
            app = row["app"]
            invocations = int(row["target"])
            active = invocations > 0

            was_warm = app in warm_apps

            # One cold-start event per active app-window
            # that was not prewarmed.
            cold_start = int(active and not was_warm)

            # A cold request loads a container temporarily.
            # With full capacity, this temporary instance
            # is assumed to be released after the window.
            temporary_load = cold_start

            records.append({
                "policy": policy_name,
                "window": window,
                "app": app,
                "invocations": invocations,
                "warm": int(was_warm),
                "cold_start_event": cold_start,
                "cold_start_penalty_ms": (
                    cold_start * COLD_START_PENALTY_MS
                ),
                "prewarm_operation": int(
                    app in prewarmed
                ),
                "eviction_operation": int(
                    app in evicted
                ),
                "temporary_cold_load": temporary_load,
            })

    details = pd.DataFrame(records)

    active_windows = int(
        (details["invocations"] > 0).sum()
    )

    cold_starts = int(
        details["cold_start_event"].sum()
    )

    summary = {
        "policy": policy_name,
        "active_app_windows": active_windows,
        "simulated_cold_start_events": cold_starts,
        "simulated_cold_start_rate": (
            cold_starts / active_windows
            if active_windows else 0
        ),
        "avg_warm_apps": (
            details.groupby("window")["warm"]
            .sum().mean()
        ),
        "prewarm_operations": int(
            details["prewarm_operation"].sum()
        ),
        "eviction_operations": int(
            details["eviction_operation"].sum()
        ),
        "temporary_cold_loads": int(
            details["temporary_cold_load"].sum()
        ),
        "total_penalty_ms": int(
            details["cold_start_penalty_ms"].sum()
        ),
    }

    return summary, details


summaries = []
details_list = []

for name, column in POLICIES.items():
    summary, details = simulate(df, name, column)
    summaries.append(summary)
    details_list.append(details)

summary_df = pd.DataFrame(summaries)
details_df = pd.concat(details_list, ignore_index=True)

os.makedirs("models", exist_ok=True)

summary_df.to_csv(
    "models/simulator_v2_summary.csv",
    index=False
)

details_df.to_csv(
    "models/simulator_v2_details.csv",
    index=False
)

print("\n========== SIMULATOR V2 RESULTS ==========")
print(summary_df.to_string(index=False))
print("==========================================")

print("\nSaved:")
print("- models/simulator_v2_summary.csv")
print("- models/simulator_v2_details.csv")
print("\nSimulator V2 complete!")

