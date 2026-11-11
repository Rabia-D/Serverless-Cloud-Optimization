
import os
import pandas as pd

# ==========================================
# SERVERLESS SIMULATOR V3
# Stateful container lifecycle simulation
# ==========================================

DATA_PATH = "models/policy_comparison_details.csv"
MAX_WARM_APPS = 4
IDLE_TIMEOUT = 3
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

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df.duplicated(["app", "window"]).any():
    raise ValueError("Duplicate app-window records")


def simulate(data, policy_name, policy_column):

    # app -> last window in which it received demand
    warm = {}

    records = []

    total_cold = 0
    total_evictions = 0
    total_loads = 0
    total_active = 0

    for window, group in data.groupby("window", sort=True):

        # Step 1: Evict applications that have
        # been idle beyond the retention timeout.
        expired = [
            app for app, last_used in warm.items()
            if window - last_used > IDLE_TIMEOUT
        ]

        for app in expired:
            del warm[app]
            total_evictions += 1

        # Step 2: Policy selects preferred warm apps.
        selected = set(
            group.loc[
                group[policy_column] == 1, "app"
            ]
        )

        if len(selected) > MAX_WARM_APPS:
            raise ValueError("Capacity exceeded")

        # Step 3: Prewarm selected applications.
        for app in sorted(selected):

            if app in warm:
                continue

            if len(warm) >= MAX_WARM_APPS:
                # Evict an unselected app first.
                candidates = [
                    a for a in warm
                    if a not in selected
                ]

                if not candidates:
                    candidates = list(warm)

                victim = min(
                    candidates,
                    key=lambda a: (warm[a], a)
                )

                del warm[victim]
                total_evictions += 1

            warm[app] = window
            total_loads += 1

        # Step 4: Process demand.
        for _, row in group.iterrows():

            app = row["app"]
            count = int(row["target"])

            active = count > 0
            cold = 0

            if active:
                total_active += 1

                if app not in warm:
                    cold = 1
                    total_cold += 1

                    if len(warm) >= MAX_WARM_APPS:
                        # Evict least recently used.
                        victim = min(
                            warm,
                            key=lambda a: (warm[a], a)
                        )
                        del warm[victim]
                        total_evictions += 1

                    warm[app] = window
                    total_loads += 1

                else:
                    warm[app] = window

            records.append({
                "policy": policy_name,
                "window": window,
                "app": app,
                "invocations": count,
                "active": int(active),
                "cold_start": cold,
                "warm_after": int(app in warm),
                "cold_penalty_ms": (
                    cold * COLD_START_PENALTY_MS
                )
            })

    details = pd.DataFrame(records)

    summary = {
        "policy": policy_name,
        "active_app_windows": total_active,
        "simulated_cold_starts": total_cold,
        "cold_start_rate": (
            total_cold / total_active
            if total_active else 0
        ),
        "total_loads": total_loads,
        "total_evictions": total_evictions,
        "total_penalty_ms": (
            total_cold * COLD_START_PENALTY_MS
        )
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
    "models/simulator_v3_summary.csv",
    index=False
)

details_df.to_csv(
    "models/simulator_v3_details.csv",
    index=False
)

print("\n========== SIMULATOR V3 RESULTS ==========")
print(summary_df.to_string(index=False))
print("==========================================")

print("\nSaved:")
print("- models/simulator_v3_summary.csv")
print("- models/simulator_v3_details.csv")
print("\nSimulator V3 complete!")

