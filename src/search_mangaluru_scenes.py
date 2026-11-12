import requests
import geopandas as gpd
from shapely.geometry import shape


BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

BOUNDARY_FILE = (
    BASE_DIR
    + r"\data\boundary\mangaluru_taluk.gpkg"
)

CATALOGUE_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)


# ---------------------------------------------------------
# Load Mangaluru boundary
# ---------------------------------------------------------

boundary = gpd.read_file(BOUNDARY_FILE)

boundary_wgs84 = boundary.to_crs("EPSG:4326")
boundary_utm = boundary.to_crs("EPSG:32643")

aoi_wgs84 = boundary_wgs84.geometry.iloc[0]
aoi_utm = boundary_utm.geometry.iloc[0]

aoi_wkt = aoi_wgs84.wkt


# ---------------------------------------------------------
# Search function
# ---------------------------------------------------------

def search_year(year, start_date, end_date):

    print("\n" + "=" * 75)
    print(f"SEARCHING SENTINEL-2 L2A FOR {year}")
    print("=" * 75)

    spatial_filter = (
        "OData.CSC.Intersects("
        "area=geography'SRID=4326;"
        + aoi_wkt
        + "')"
    )

    collection_filter = (
        "Collection/Name eq 'SENTINEL-2'"
    )

    product_type_filter = (
        "Attributes/OData.CSC.StringAttribute/"
        "any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
    )

    cloud_filter = (
        "Attributes/OData.CSC.DoubleAttribute/"
        "any(att:att/Name eq 'cloudCover' "
        "and att/OData.CSC.DoubleAttribute/Value le 30.0)"
    )

    date_filter = (
        f"ContentDate/Start gt {start_date}T00:00:00.000Z "
        f"and ContentDate/Start lt {end_date}T23:59:59.999Z"
    )

    filter_query = (
        collection_filter
        + " and "
        + product_type_filter
        + " and "
        + cloud_filter
        + " and "
        + spatial_filter
        + " and "
        + date_filter
    )

    params = {
        "$filter": filter_query,
        "$top": 100,
        "$expand": "Attributes"
    }

    print("\nQuerying Copernicus catalogue...")

    try:

        response = requests.get(
            CATALOGUE_URL,
            params=params,
            timeout=90
        )

        print("HTTP status:", response.status_code)

        if response.status_code != 200:

            print("\nServer response:")
            print(response.text[:2000])
            return

        data = response.json()

    except Exception as e:

        print("\nRequest failed:")
        print(type(e).__name__, e)
        return

    products = data.get("value", [])

    print("Products returned:", len(products))

    candidates = []

    for product in products:

        geo = product.get("GeoFootprint")

        if not geo:
            continue

        try:
            footprint_wgs84 = shape(geo)

            footprint_utm = (
                gpd.GeoSeries(
                    [footprint_wgs84],
                    crs="EPSG:4326"
                )
                .to_crs("EPSG:32643")
                .iloc[0]
            )

            intersection = aoi_utm.intersection(
                footprint_utm
            )

            coverage = (
                intersection.area
                / aoi_utm.area
            ) * 100

        except Exception as e:

            print(
                "Could not calculate coverage for:",
                product.get("Name")
            )
            print(e)
            continue

        cloud_cover = None

        for attribute in product.get("Attributes", []):

            if attribute.get("Name") == "cloudCover":

                cloud_cover = attribute.get(
                    "Value"
                )

                break

        candidates.append(
            {
                "name": product.get("Name"),
                "id": product.get("Id"),
                "date": product.get(
                    "ContentDate",
                    {}
                ).get("Start"),
                "cloud": cloud_cover,
                "coverage": coverage
            }
        )

    # Highest Mangaluru coverage first,
    # then lowest cloud cover.
    candidates.sort(
        key=lambda x: (
            -x["coverage"],
            float(x["cloud"])
            if x["cloud"] is not None
            else 999
        )
    )

    print("\nCANDIDATE SCENES")
    print("-" * 75)

    if not candidates:

        print("No candidate scenes found.")
        return

    for i, candidate in enumerate(
        candidates[:20],
        start=1
    ):

        print(f"\n{i}.")
        print("Product:", candidate["name"])
        print("ID:", candidate["id"])
        print("Date:", candidate["date"])
        print(
            "Cloud cover:",
            candidate["cloud"],
            "%"
        )
        print(
            "Mangaluru coverage:",
            round(candidate["coverage"], 2),
            "%"
        )


# ---------------------------------------------------------
# Search periods
# ---------------------------------------------------------

search_year(
    2022,
    "2022-11-01",
    "2023-02-28"
)

search_year(
    2025,
    "2025-11-01",
    "2026-02-28"
)

print("\nSearch completed.")