import pandas as pd
from pathlib import Path

path = Path("data/results/crop_reference/mangaluru_crop_reference_points.csv")
df = pd.read_csv(path)

paddy = df[df["crop_class"].astype(str).str.strip().str.lower() == "paddy"].copy()

paddy["lat_key"] = pd.to_numeric(paddy["latitude"], errors="coerce").round(5)
paddy["lon_key"] = pd.to_numeric(paddy["Longtitude"], errors="coerce").round(5)

print("Paddy reference records:", len(paddy))
print("Unique coordinate groups (5 decimal places):",
      paddy[["lat_key", "lon_key"]].drop_duplicates().shape[0])

print("\nRecords and unique locations by village:")
summary = paddy.groupby("village").agg(
    records=("crop_class", "size"),
    unique_locations=("lat_key", lambda s: paddy.loc[s.index, ["lat_key", "lon_key"]].drop_duplicates().shape[0])
)
print(summary.to_string())

print("\nMost repeated coordinate groups:")
counts = paddy.groupby(["lat_key", "lon_key"]).size().sort_values(ascending=False)
print(counts.head(10).to_string())
