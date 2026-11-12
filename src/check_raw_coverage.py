import os
import glob
import rasterio
from rasterio.windows import Window


BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

for year in [2018, 2022, 2025]:

    pattern = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        str(year),
        "**",
        "*_B02_10m.jp2"
    )

    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"\n{year}: B02 not found")
        continue

    b02 = files[0]

    print(f"\n{'=' * 60}")
    print(f"YEAR: {year}")
    print("FILE:", b02)

    with rasterio.open(b02) as src:

        print("CRS:", src.crs)
        print("Full bounds:", src.bounds)
        print("Size:", src.width, "x", src.height)
        print("NoData:", src.nodata)

        min_row = src.height
        min_col = src.width
        max_row = -1
        max_col = -1
        valid_count = 0

        for _, window in src.block_windows(1):

            data = src.read(1, window=window)

            valid = data > 0

            if not valid.any():
                continue

            rows, cols = valid.nonzero()

            global_rows = rows + int(window.row_off)
            global_cols = cols + int(window.col_off)

            min_row = min(min_row, global_rows.min())
            max_row = max(max_row, global_rows.max())
            min_col = min(min_col, global_cols.min())
            max_col = max(max_col, global_cols.max())

            valid_count += int(valid.sum())

        print("Valid pixels:", valid_count)

        if valid_count > 0:

            valid_window = Window(
                min_col,
                min_row,
                max_col - min_col + 1,
                max_row - min_row + 1
            )

            print(
                "Valid-data bounds:",
                src.window_bounds(valid_window)
            )

        else:
            print("No valid pixels found.")


print("\nCoverage check completed.")