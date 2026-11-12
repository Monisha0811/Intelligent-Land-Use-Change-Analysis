import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LAND USE CHANGE VISUALIZATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 2018 vs 2022
# ============================================================

df_2018_2022 = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "area_change_2018_2022.csv"
    )
)

print("\nCreating 2018 vs 2022 area chart...")

x = range(len(df_2018_2022))

plt.figure(figsize=(10, 6))

plt.bar(
    [i - 0.2 for i in x],
    df_2018_2022["2018_Area_km2"],
    width=0.4,
    label="2018"
)

plt.bar(
    [i + 0.2 for i in x],
    df_2018_2022["2022_Area_km2"],
    width=0.4,
    label="2022"
)

plt.xticks(
    x,
    df_2018_2022["Class"],
    rotation=20
)

plt.xlabel("Land-use class")
plt.ylabel("Area (km²)")
plt.title("Land-use Area Comparison: 2018 vs 2022")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "area_comparison_2018_2022.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. 2022 vs 2025
# ============================================================

df_2022_2025 = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "area_change_2022_2025.csv"
    )
)

print("Creating 2022 vs 2025 area chart...")

x = range(len(df_2022_2025))

plt.figure(figsize=(10, 6))

plt.bar(
    [i - 0.2 for i in x],
    df_2022_2025["2022_Area_km2"],
    width=0.4,
    label="2022"
)

plt.bar(
    [i + 0.2 for i in x],
    df_2022_2025["2025_Area_km2"],
    width=0.4,
    label="2025"
)

plt.xticks(
    x,
    df_2022_2025["Class"],
    rotation=20
)

plt.xlabel("Land-use class")
plt.ylabel("Area (km²)")
plt.title("Land-use Area Comparison: 2022 vs 2025")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "area_comparison_2022_2025.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 3. 2018 vs 2025
# ============================================================

df_2018_2025 = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "area_change_2018_2025.csv"
    )
)

print("Creating 2018 vs 2025 area chart...")

x = range(len(df_2018_2025))

plt.figure(figsize=(10, 6))

plt.bar(
    [i - 0.2 for i in x],
    df_2018_2025["2018_Area_km2"],
    width=0.4,
    label="2018"
)

plt.bar(
    [i + 0.2 for i in x],
    df_2018_2025["2025_Area_km2"],
    width=0.4,
    label="2025"
)

plt.xticks(
    x,
    df_2018_2025["Class"],
    rotation=20
)

plt.xlabel("Land-use class")
plt.ylabel("Area (km²)")
plt.title("Land-use Area Comparison: 2018 vs 2025")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "area_comparison_2018_2025.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 4. NET CHANGE 2018 → 2025
# ============================================================

print("Creating net change chart...")

plt.figure(figsize=(10, 6))

plt.bar(
    df_2018_2025["Class"],
    df_2018_2025["Change_km2"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("Land-use class")
plt.ylabel("Change in area (km²)")
plt.title("Net Land-use Change: 2018 → 2025")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "net_change_2018_2025.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 5. LAND-USE TREND 2018–2025
# ============================================================

print("Creating land-use trend chart...")

years = [2018, 2022, 2025]

plt.figure(figsize=(11, 7))

classes = df_2018_2025["Class"].tolist()

for class_name in classes:

    row_2018 = df_2018_2022[
        df_2018_2022["Class"] == class_name
    ]

    row_2022 = df_2022_2025[
        df_2022_2025["Class"] == class_name
    ]

    row_2025 = df_2018_2025[
        df_2018_2025["Class"] == class_name
    ]

    values = [
        row_2018.iloc[0]["2018_Area_km2"],
        row_2022.iloc[0]["2022_Area_km2"],
        row_2025.iloc[0]["2025_Area_km2"]
    ]

    plt.plot(
        years,
        values,
        marker="o",
        linewidth=2,
        label=class_name
    )


plt.xlabel("Year")
plt.ylabel("Area (km²)")
plt.title("Land-use Area Trend: 2018–2025")

plt.xticks(years)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "land_use_trend_2018_2025.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 65)
print("VISUALIZATION CREATION COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nCharts saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", filename)