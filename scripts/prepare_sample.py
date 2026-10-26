import pandas as pd

FILE = "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"

# Read the full trace in chunks to avoid loading it all into memory.
counts = {}

for chunk in pd.read_csv(FILE, chunksize=100_000):
    chunk["window"] = (
        chunk["end_timestamp"] // 300
    ).astype("int64")

    grouped = chunk.groupby(["app", "window"]).size()

    for key, count in grouped.items():
        counts[key] = counts.get(key, 0) + int(count)

# Convert aggregated data to a DataFrame.
df = pd.DataFrame(
    [(app, window, count) for (app, window), count in counts.items()],
    columns=["app", "window", "invocations"]
)

# Select the 10 applications with the most invocations.
top_apps = (
    df.groupby("app")["invocations"]
    .sum()
    .nlargest(10)
    .index
)

sample = df[df["app"].isin(top_apps)].copy()
sample["window_start_seconds"] = sample["window"] * 300

sample = sample.sort_values(["app", "window"])

sample.to_csv("azure_2021_sample.csv", index=False)

print("Sample created!")
print("Applications:", sample["app"].nunique())
print("Rows:", len(sample))
print(sample.head(10))
