import pandas as pd
import matplotlib.pyplot as plt

# Load the processed Azure 2021 dataset
df = pd.read_csv("data/azure_2021_sample_complete.csv")

# Identify the most bursty and a lower-activity application
summary = df.groupby("app")["invocations"].agg(
    ["sum", "max", "mean"]
)

bursty_app = summary["max"].idxmax()
lower_app = summary["sum"].idxmin()

# Plot the first three days of the trace
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

for ax, app, label in zip(
    axes,
    [bursty_app, lower_app],
    ["Highest observed spike", "Lowest total invocations"]
):
    subset = df[
        (df["app"] == app) &
        (df["window"] < 864)
    ].sort_values("window")

    ax.plot(
        subset["window_start_seconds"] / 86400,
        subset["invocations"],
        linewidth=0.9
    )

    ax.set_title(label)
    ax.set_ylabel("Invocations per 5 minutes")
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Days since trace start")

fig.suptitle(
    "Azure Functions 2021: Application Invocation Patterns",
    fontsize=14
)

plt.tight_layout()
plt.savefig(
    "figures/azure_invocation_patterns.png",
    dpi=200,
    bbox_inches="tight"
)

print("Graph saved: figures/azure_invocation_patterns.png")
print("Highest-spike app:", bursty_app)
print("Lowest-total app:", lower_app)
