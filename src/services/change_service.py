"""
Dynamic Land Use Change Detection Service (Phase 4).

Compares any two classified LULC rasters, generates spatial change maps,
computes land-use transition matrices, calculates class-wise area changes,
and highlights critical transitions (e.g. Agriculture -> Built-up).
"""

import os
import re
import csv
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except (ImportError, Exception):
    rasterio = None
    HAS_RASTERIO = False

from src.config import (
    BASE_DIR,
    CLASSIFIED_MANGALURU_DIR,
    CHANGE_MAPS_DIR,
    MAPS_DIR,
    RESULTS_DIR,
    LULC_CLASSES,
    PIXEL_AREA_KM2,
)

logger = logging.getLogger("change_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [ChangeService] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ChangeService:
    """Service handling multi-temporal land use change detection and transition dynamics."""

    def __init__(self):
        pass

    def list_available_years(self) -> List[str]:
        """
        Scan data/classified/mangaluru for land_use_<year>.tif files
        and return sorted list of available years.
        """
        pattern = CLASSIFIED_MANGALURU_DIR / "land_use_*.tif"
        files = glob_files = list(CLASSIFIED_MANGALURU_DIR.glob("land_use_*.tif"))

        years = []
        for f in files:
            # Match e.g. land_use_2022.tif -> 2022
            match = re.match(r"^land_use_([A-Za-z0-9_-]+)\.tif$", f.name)
            if match:
                years.append(match.group(1))

        # Sort chronologically if numeric, otherwise alphabetically
        def sort_key(val):
            return (0, int(val)) if val.isdigit() else (1, val)

        years.sort(key=sort_key)
        return years

    def get_available_periods(self) -> List[str]:
        """
        Return list of standard and dynamic comparison periods.
        """
        years = self.list_available_years()
        periods = []

        # Standard historical periods first if available
        standard_periods = ["2018-2022", "2022-2025", "2018-2025"]
        for sp in standard_periods:
            y1, y2 = sp.split("-")
            if y1 in years and y2 in years:
                periods.append(sp)

        # Add any newly available pairs
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                p = f"{years[i]}-{years[j]}"
                if p not in periods:
                    periods.append(p)

        return periods

    def load_raster_data(self, year: Union[str, int]) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Load classified LULC raster for a given year.
        Uses rasterio if available, otherwise falls back to PIL.
        """
        raster_path = CLASSIFIED_MANGALURU_DIR / f"land_use_{year}.tif"
        if not raster_path.exists():
            raise FileNotFoundError(f"Classified raster for year {year} not found: {raster_path}")

        profile = None
        if HAS_RASTERIO and rasterio is not None:
            with rasterio.open(str(raster_path)) as src:
                data = src.read(1)
                profile = src.profile.copy()
        else:
            img = Image.open(str(raster_path))
            data = np.array(img, dtype=np.uint8)

        return data, profile

    def run_change_detection(
        self,
        year1: Union[str, int],
        year2: Union[str, int],
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Run pixel-by-pixel change detection between year1 and year2.
        Generates:
        1. Spatial change raster: data/change_maps/change_{year1}_{year2}.tif
        2. Change map image: outputs/maps/change_map_{year1}_{year2}.png
        3. Area change statistics: data/results/area_change_{year1}_{year2}.csv
        4. Transition matrix: data/results/transition_matrix_{year1}_{year2}.csv
        5. Detailed transitions: data/results/transition_details_{year1}_{year2}.csv
        """
        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        def check_abort():
            if is_cancelled and is_cancelled():
                raise RuntimeError("Change detection cancelled by user.")

        year1_str, year2_str = str(year1).strip(), str(year2).strip()

        if year1_str == year2_str:
            raise ValueError(f"Cannot compare a year with itself ({year1_str} -> {year2_str}).")

        report(10, f"Loading classified maps for {year1_str} and {year2_str}...")
        check_abort()

        map1, profile1 = self.load_raster_data(year1_str)
        map2, profile2 = self.load_raster_data(year2_str)

        if map1.shape != map2.shape:
            raise ValueError(
                f"Raster dimensions do not match for {year1_str} and {year2_str}: "
                f"{map1.shape} vs {map2.shape}."
            )

        report(25, "Identifying common valid pixels...")
        check_abort()

        # Valid pixels: must be classified in BOTH years (classes 1 to 5)
        valid = (
            (map1 >= 1) & (map1 <= 5) &
            (map2 >= 1) & (map2 <= 5)
        )

        valid_pixels = int(np.sum(valid))
        if valid_pixels == 0:
            raise ValueError(f"No common valid classified pixels found between {year1_str} and {year2_str}.")

        report(40, "Computing spatial change categories (unchanged vs changed)...")
        check_abort()

        unchanged = valid & (map1 == map2)
        changed = valid & (map1 != map2)

        unchanged_pixels = int(np.sum(unchanged))
        changed_pixels = int(np.sum(changed))

        changed_pct = round((changed_pixels / valid_pixels) * 100.0, 2)
        unchanged_pct = round((unchanged_pixels / valid_pixels) * 100.0, 2)

        # Create change map: 0=NoData, 1=Unchanged, 2=Changed
        change_map = np.zeros(map1.shape, dtype=np.uint8)
        change_map[unchanged] = 1
        change_map[changed] = 2

        # -----------------------------------------------------------------
        # 1. Save Spatial Change Raster
        # -----------------------------------------------------------------
        report(55, "Saving spatial change raster...")
        check_abort()

        change_raster_path = CHANGE_MAPS_DIR / f"change_{year1_str}_{year2_str}.tif"
        if HAS_RASTERIO and rasterio is not None and profile1 is not None:
            profile = profile1.copy()
            profile.update(
                dtype="uint8",
                count=1,
                compress="lzw",
                nodata=0
            )
            with rasterio.open(str(change_raster_path), "w", **profile) as dst:
                dst.write(change_map, 1)
                dst.set_band_description(1, "0=NoData, 1=Unchanged, 2=Changed")
        else:
            pil_change = Image.fromarray(change_map)
            pil_change.save(str(change_raster_path), format="TIFF", compression="tiff_lzw")

        # -----------------------------------------------------------------
        # 2. Save Change Map PNG
        # -----------------------------------------------------------------
        report(65, "Rendering change map image...")
        check_abort()

        change_png_path = MAPS_DIR / f"change_map_{year1_str}_{year2_str}.png"
        rgb = np.zeros((map1.shape[0], map1.shape[1], 3), dtype=np.uint8)
        rgb[change_map == 0] = [255, 255, 255]  # NoData = white
        rgb[change_map == 1] = [220, 220, 220]  # Unchanged = light gray
        rgb[change_map == 2] = [220, 50, 50]    # Changed = crimson red

        Image.fromarray(rgb).save(str(change_png_path))

        # -----------------------------------------------------------------
        # 3. Class Area Change Analysis
        # -----------------------------------------------------------------
        report(75, "Computing class-wise area changes and percentage transitions...")
        check_abort()

        area_change_rows = []
        for class_id, class_name in LULC_CLASSES.items():
            count1 = int(np.sum((map1 == class_id) & valid))
            count2 = int(np.sum((map2 == class_id) & valid))

            area1 = round(count1 * PIXEL_AREA_KM2, 2)
            area2 = round(count2 * PIXEL_AREA_KM2, 2)
            diff_km2 = round(area2 - area1, 2)

            pct_change = round(((diff_km2 / area1) * 100.0), 2) if area1 > 0 else 0.0

            area_change_rows.append({
                "Class": class_name,
                f"{year1_str}_Area_km2": area1,
                f"{year2_str}_Area_km2": area2,
                "Change_km2": diff_km2,
                "Percentage_Change": pct_change
            })

        area_change_df = pd.DataFrame(area_change_rows)
        area_csv_path = RESULTS_DIR / f"area_change_{year1_str}_{year2_str}.csv"
        area_change_df.to_csv(str(area_csv_path), index=False)

        # -----------------------------------------------------------------
        # 4. Land Use Transition Matrix (5x5)
        # -----------------------------------------------------------------
        report(85, "Calculating 5x5 transition matrix...")
        check_abort()

        matrix_counts = np.zeros((5, 5), dtype=np.int64)
        old_vals = map1[valid]
        new_vals = map2[valid]

        for old_c in range(1, 6):
            for new_c in range(1, 6):
                count = int(np.sum((old_vals == old_c) & (new_vals == new_c)))
                matrix_counts[old_c - 1, new_c - 1] = count

        transition_df = pd.DataFrame(
            matrix_counts,
            index=[f"{name} ->" for name in LULC_CLASSES.values()],
            columns=list(LULC_CLASSES.values())
        )
        matrix_csv_path = RESULTS_DIR / f"transition_matrix_{year1_str}_{year2_str}.csv"
        transition_df.to_csv(str(matrix_csv_path))

        # -----------------------------------------------------------------
        # 5. Detailed Transitions & Critical Trajectories
        # -----------------------------------------------------------------
        report(92, "Identifying critical transitions (e.g. Agriculture -> Built-up)...")
        check_abort()

        detail_rows = []
        critical_transitions = []

        for old_c in range(1, 6):
            for new_c in range(1, 6):
                if old_c == new_c:
                    continue  # Unchanged

                count = int(matrix_counts[old_c - 1, new_c - 1])
                if count > 0:
                    area = round(count * PIXEL_AREA_KM2, 2)
                    from_name = LULC_CLASSES[old_c]
                    to_name = LULC_CLASSES[new_c]

                    row = {
                        "From": from_name,
                        "To": to_name,
                        "Pixels": count,
                        "Area_km2": area
                    }
                    detail_rows.append(row)

                    # Highlight key transitions of environmental concern
                    if from_name in ["Agriculture", "Vegetation/Forest", "Barren/Open"] and to_name == "Built-up":
                        critical_transitions.append({
                            "transition": f"{from_name} -> {to_name}",
                            "area_km2": area,
                            "pixels": count,
                            "type": "Urbanization / Built-up Expansion"
                        })

        # Sort detail rows by area descending
        detail_rows.sort(key=lambda x: -x["Area_km2"])
        transition_details_df = pd.DataFrame(detail_rows)
        details_csv_path = RESULTS_DIR / f"transition_details_{year1_str}_{year2_str}.csv"
        transition_details_df.to_csv(str(details_csv_path), index=False)

        report(100, f"Change detection between {year1_str} and {year2_str} completed successfully.")

        return {
            "success": True,
            "period": f"{year1_str}-{year2_str}",
            "year1": year1_str,
            "year2": year2_str,
            "total_pixels": int(map1.size),
            "valid_pixels": valid_pixels,
            "changed_pixels": changed_pixels,
            "unchanged_pixels": unchanged_pixels,
            "changed_percentage": changed_pct,
            "unchanged_percentage": unchanged_pct,
            "change_raster": str(change_raster_path),
            "change_map_png": str(change_png_path),
            "area_change": area_change_rows,
            "transition_details": detail_rows,
            "critical_transitions": critical_transitions
        }


# Global singleton instance
change_service = ChangeService()
