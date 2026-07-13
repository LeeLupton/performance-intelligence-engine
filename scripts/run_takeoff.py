"""Run the Ocean Reef blade-edge takeoff: classify edges, sum LF.

All length math in EPSG:2264 (NC StatePlane ft).
"""

import os
import sys

import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import manual_layers
from edge_classify import ClassifyInputs, classify

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data")


def main():
    pav = gpd.read_file(os.path.join(DATA, "pavement_ccap_2264.geojson"))
    bldg = gpd.read_file(os.path.join(DATA, "buildings_2264.geojson"))
    diss = gpd.read_file(os.path.join(DATA, "parcel_scope_dissolved.geojson"))
    scope_full = diss[diss.scope == "core_plus_extended"].geometry.iloc[0]
    scope_core = diss[diss.scope == "core_rule"].geometry.iloc[0]

    inp = ClassifyInputs(
        pavement=pav,
        scope=scope_full,
        buildings=bldg,
        dune_zone=manual_layers.dune_zone(),
        natural_zones=list(manual_layers.NATURAL_POLYS),
        bed_interior_zones=list(manual_layers.BED_INTERIOR_POLYS),
        farside_zone=manual_layers.farside_zone(),
        verge_zone=manual_layers.row_verge_zone(),
        type_zones=manual_layers.all_type_zones(),
    )
    res = classify(inp)
    edges = res["edge_candidates"]

    # manual landscape-bed lines (BLADE per operator spec; beds don't
    # appear in impervious data)
    beds = gpd.GeoDataFrame(
        [{"class": "BLADE", "feature_type": "bed", "source": "manual",
          "geometry": g} for _, _, g in manual_layers.BED_LINES],
        crs="EPSG:2264")
    beds["length_ft"] = beds.geometry.length
    edges = gpd.GeoDataFrame(pd.concat([edges, beds], ignore_index=True),
                             crs="EPSG:2264")

    edges.to_file(os.path.join(DATA, "edge_candidates_2264.geojson"),
                  driver="GeoJSON")
    res["pavement"].to_file(os.path.join(DATA, "pavement_clean_2264.geojson"),
                            driver="GeoJSON")

    pd.set_option("display.width", 200)
    summary = (edges.groupby(["class", "feature_type"])["length_ft"]
               .agg(["sum", "count"]).round(1))
    print(summary.to_string())
    blade = edges[edges["class"] == "BLADE"]
    print(f"\nTOTAL BLADE LF (core+extended scope): "
          f"{blade.length_ft.sum():,.0f}")

    core_clip = blade.geometry.intersection(scope_core.buffer(
        30))  # frontage allowance parity
    core_lf = sum(g.length for g in core_clip if not g.is_empty)
    print(f"BLADE LF within core-rule scope (+30 ft frontage): {core_lf:,.0f}")
    by_type = blade.groupby("feature_type").length_ft.sum().round(0)
    print("\nBLADE by feature_type:")
    print(by_type.to_string())
    unclassified = edges["class"].isna().sum()
    print(f"\nunclassified segments: {unclassified}")


if __name__ == "__main__":
    main()
