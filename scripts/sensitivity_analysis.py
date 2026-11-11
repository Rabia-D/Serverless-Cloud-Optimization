
import pandas as pd
import itertools
import os

DATA = "models/policy_comparison_details.csv"

df = pd.read_csv(DATA)
df = df.sort_values(["window", "app"])

POLICIES = {
    "Static": "static_warm",
    "Recent activity": "recent_warm",
    "ML prediction": "ml_warm"
}

CAPACITIES = [2, 4, 6]
TIMEOUTS = [1, 3, 6]

results = []


def simulate(data, policy_col, capacity, timeout):
    warm = {}
    total_cold = 0
    total_active = 0
    total_warm = 0

    for window, group in data.groupby("window"):

        # Expire idle containers.
        expired = [
            app for app, last_used in warm.items()
            if window - last_used > timeout
        ]

        for app in expired:
            del warm[app]

        # Select policy-preferred applications.
        selected_rows = group[
            group[policy_col] == 1
        ].copy()

        # If capacity is smaller than the
        # original four-slot policy, truncate.
        selected_rows = selected_rows.head(capacity)
        selected = set(selected_rows["app"])

        # Prewarm selected applications.
        for app in sorted(selected):
            if app in warm:
                continue

            if len(warm) >= capacity:
                candidates = [
                    a for a in warm
                    if a not in selected
                ]

                if not candidates:
                    raise RuntimeError("No eviction candidate")

                victim = min(
                    candidates,
                    key=lambda a: (warm[a], a)
                )
                del warm[victim]

            warm[app] = window

        # Evaluate all demand against one snapshot.
        warm_before = set(warm)

        active = set(
            group.loc[group["target"] > 0, "app"]
        )

        cold = active - warm_before

        total_active += len(active)
        total_cold += len(cold)

        # Retain policy-selected containers.
        next_warm = {
            app: warm.get(app, window)
            for app in selected
        }

        for app in selected & active:
            next_warm[app] = window

        spare = capacity - len(next_warm)

        # Retain active nonselected apps.
        other_active = sorted(
            active - selected,
            key=lambda a: (
                a not in warm_before, a
            )
        )

        for app in other_active[:spare]:
            next_warm[app] = window

        spare = capacity - len(next_warm)

        # Retain other warm apps if space remains.
        remaining = sorted(
            set(warm_before) - set(next_warm),
            key=lambda a: (-warm[a], a)
        )

        for app in remaining[:spare]:
            next_warm[app] = warm[app]

        warm = next_warm

        if len(warm) > capacity:
            raise RuntimeError("Capacity exceeded")

        total_warm += len(warm)

    return {
        "cold_start_rate": (
            total_cold / total_active
            if total_active else 0
        ),
        "simulated_cold_starts": total_cold,
        "avg_warm_apps": (
            total_warm / data["window"].nunique()
        )
    }


for capacity, timeout in itertools.product(
    CAPACITIES, TIMEOUTS
):
    for name, column in POLICIES.items():
        metrics = simulate(
            df, column, capacity, timeout
        )

        results.append({
            "policy": name,
            "capacity": capacity,
            "idle_timeout": timeout,
            **metrics
        })

summary = pd.DataFrame(results)

os.makedirs("models", exist_ok=True)

summary.to_csv(
    "models/sensitivity_results.csv",
    index=False
)

print("\n===== SENSITIVITY ANALYSIS =====")
print(summary.to_string(index=False))
print("\nSaved: models/sensitivity_results.csv")

