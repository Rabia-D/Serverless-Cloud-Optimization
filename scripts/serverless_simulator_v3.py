
import os
import pandas as pd

# ==========================================
# SERVERLESS SIMULATOR V3 — CORRECTED
# Window-level stateful lifecycle simulation
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

if (df["target"] < 0).any():
    raise ValueError("Negative invocation counts")

if (df["target"].isna()).any():
    raise ValueError("Missing invocation counts")


def simulate(data, policy_name, policy_column):

    # app -> most recent activity window
    warm = {}

    records = []
    total_cold = 0
    total_loads = 0
    total_evictions = 0
    total_prewarm = 0
    total_active = 0
    total_warm_slots = 0

    windows = data["window"].nunique()

    for window, group in data.groupby("window", sort=True):

        # ----------------------------------
        # 1. EXPIRE IDLE CONTAINERS
        # ----------------------------------

        expired = [
            app for app, last_used in warm.items()
            if window - last_used > IDLE_TIMEOUT
        ]

        for app in expired:
            del warm[app]
            total_evictions += 1

        # ----------------------------------
        # 2. APPLY POLICY DECISIONS
        # ----------------------------------

        selected = set(
            group.loc[
                group[policy_column] == 1, "app"
            ]
        )

        if len(selected) > MAX_WARM_APPS:
            raise ValueError("Policy exceeds capacity")

        # Evict unselected apps only when
        # capacity is needed for selected apps.
        for app in sorted(selected):

            if app in warm:
                continue

            if len(warm) >= MAX_WARM_APPS:

                candidates = [
                    a for a in warm
                    if a not in selected
                ]

                if not candidates:
                    raise RuntimeError(
                        "No evictable container available"
                    )

                victim = min(
                    candidates,
                    key=lambda a: (warm[a], a)
                )

                del warm[victim]
                total_evictions += 1

            warm[app] = window
            total_loads += 1
            total_prewarm += 1

        # ----------------------------------
        # 3. SNAPSHOT STATE BEFORE DEMAND
        # ----------------------------------

        warm_before_demand = set(warm)

        # All applications in this window are
        # evaluated against the SAME snapshot.
        # This avoids arbitrary row-order effects.
        active_apps = set(
            group.loc[group["target"] > 0, "app"]
        )

        cold_apps = active_apps - warm_before_demand

        total_active += len(active_apps)
        total_cold += len(cold_apps)

        # ----------------------------------
        # 4. UPDATE CONTAINER STATE
        # ----------------------------------

        # Selected warm apps are protected.
        # Cold apps may remain warm if space exists.
        #
        # At most four containers can be retained.
        # Prefer policy-selected apps, then
        # recently active nonselected apps.

        next_warm = {
            app: warm.get(app, window)
            for app in selected
        }

        # Update last-used times for selected
        # apps that actually received demand.
        for app in selected & active_apps:
            next_warm[app] = window

        # Other active apps can occupy spare slots.
        spare_slots = MAX_WARM_APPS - len(next_warm)

        nonselected_active = active_apps - selected

        # Prefer containers already warm, then
        # deterministic application ID ordering.
        candidates = sorted(
            nonselected_active,
            key=lambda a: (
                a not in warm_before_demand,
                a
            )
        )

        for app in candidates[:spare_slots]:
            next_warm[app] = window

        # Retain other warm containers in spare
        # capacity, subject to idle timeout.
        spare_slots = MAX_WARM_APPS - len(next_warm)

        remaining = [
            app for app in warm_before_demand
            if app not in next_warm
        ]

        remaining.sort(
            key=lambda a: (-warm[a], a)
        )

        for app in remaining[:spare_slots]:
            next_warm[app] = warm[app]

        # Count state transitions at the boundary.
        removed = set(warm) - set(next_warm)
        added = set(next_warm) - set(warm)

        total_evictions += len(removed)
        total_loads += len(added)

        warm = next_warm

        if len(warm) > MAX_WARM_APPS:
            raise RuntimeError("Capacity exceeded")

        total_warm_slots += len(warm)

        # ----------------------------------
        # 5. RECORD RESULTS
        # ----------------------------------

        for _, row in group.iterrows():

            app = row["app"]
            count = int(row["target"])

            cold = int(app in cold_apps)

            records.append({
                "policy": policy_name,
                "window": window,
                "app": app,
                "invocations": count,
                "active": int(count > 0),
                "warm_before_demand": int(
                    app in warm_before_demand
                ),
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
        "avg_warm_apps": (
            total_warm_slots / windows
            if windows else 0
        ),
        "total_loads": total_loads,
        "total_evictions": total_evictions,
        "prewarm_operations": total_prewarm,
        "total_penalty_ms": (
            total_cold * COLD_START_PENALTY_MS
        )
    }

    return summary, details


summaries = []
all_details = []

for name, column in POLICIES.items():

    summary, details = simulate(
        df, name, column
    )

    summaries.append(summary)
    all_details.append(details)

summary_df = pd.DataFrame(summaries)
details_df = pd.concat(
    all_details, ignore_index=True
)

os.makedirs("models", exist_ok=True)

summary_df.to_csv(
    "models/simulator_v3_summary.csv",
    index=False
)

details_df.to_csv(
    "models/simulator_v3_details.csv",
    index=False
)

print("\n========== CORRECTED SIMULATOR V3 ==========")
print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)
print("===========================================")

print("\nSaved:")
print("- models/simulator_v3_summary.csv")
print("- models/simulator_v3_details.csv")
print("\nSimulation complete!")

