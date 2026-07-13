"""Build the Ocean Reef parcel scope from the Carteret parcel pull.

Two scope classes, kept separate for the audit trail:
  core     - brief rule: HOA common parcel + OCEAN REEF DEVELOPMENT CO
             retained tracts + adjoining Jarrick 2519 tract (9 parcels)
  extended - remaining Ocean Reef common-area land tracts (phases 3/4/5/
             13/15/17/19/CONDO/S1 deeded to unit-owner groups, plus the
             Jarrick P14 tract). These form the rest of the complex
             grounds; included in the working scope but flagged for
             client confirmation.

Outputs (EPSG:2264): data/parcel_scope_parcels.geojson (pre-dissolve),
data/parcel_scope_dissolved.geojson (core / core+extended polygons).
"""

import os

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data")

CORE_PINS = {
    "631414436510000",  # HOA common/recreation parcel, 2500 Ocean Dr, 2.10 ac
    "631414433078000",  # DevCo COMMON AREA P9
    "631414434150000",  # DevCo COMMON AREA PH8
    "631414435142000",  # DevCo COMMON AREA P7
    "631414436124000",  # DevCo COMMON AREA P6
    "631414438577000",  # DevCo COMMON AREA PIZ
    "631414439558000",  # DevCo COMMON AREA P11
    "631415530264000",  # DevCo ETAL COMMON AREA S1
    "631414432180000",  # JARRICK INC 0.72 ac at 2519 Ocean Dr (adjoins, 0 ft)
}
EXTENDED_PINS = {
    "631414433447000",  # COMMON AREA P17
    "631414433574000",  # COMMON AREA P19
    "631414434428000",  # COMMON AREA CONDO
    "631414435329000",  # COMMON AREA P15
    "631414437146000",  # COMMON AREA P5
    "631414437596000",  # COMMON AREA P13
    "631414438136000",  # COMMON AREA P4
    "631414439210000",  # COMMON AREA P3
    "631414439273000",  # COMMON AREA S1 (second tract)
    "631414437512000",  # JARRICK INC COMMON AREA P14
}


def main():
    g = gpd.read_file(os.path.join(RAW, "ocean_reef_parcels_2264.geojson"))
    if g.crs is None:
        g = g.set_crs("EPSG:2264")
    assert str(g.crs) == "EPSG:2264", g.crs

    g["scope_class"] = "reference"
    g.loc[g["PIN15"].isin(CORE_PINS), "scope_class"] = "core"
    g.loc[g["PIN15"].isin(EXTENDED_PINS), "scope_class"] = "extended"
    sel = g[g["scope_class"] != "reference"].copy()
    assert len(sel) == 19, f"expected 19 scope parcels, got {len(sel)}"

    keep = ["PIN15", "OWNER", "SITE_HOUSE", "SITE_ST", "GISacres",
            "scope_class", "geometry"]
    keep = [c for c in keep if c in sel.columns]
    sel = sel[keep]
    sel.to_file(os.path.join(OUT, "parcel_scope_parcels.geojson"),
                driver="GeoJSON")

    core_geom = unary_union(sel[sel.scope_class == "core"].geometry.values)
    full_geom = unary_union(sel.geometry.values)
    diss = gpd.GeoDataFrame(
        {"scope": ["core_rule", "core_plus_extended"],
         "parcels": [int((sel.scope_class == "core").sum()), len(sel)],
         "acres": [round(core_geom.area / 43560, 3),
                   round(full_geom.area / 43560, 3)],
         "geometry": [core_geom, full_geom]},
        crs="EPSG:2264")
    diss.to_file(os.path.join(OUT, "parcel_scope_dissolved.geojson"),
                 driver="GeoJSON")

    print(sel.groupby("scope_class")[["GISacres"]].agg(["count", "sum"]))
    print(diss[["scope", "parcels", "acres"]].to_string(index=False))
    print("full scope bounds 2264:", [round(v, 1) for v in full_geom.bounds])
    print("core contiguous parts:", len(getattr(core_geom, "geoms", [core_geom])))
    print("full contiguous parts:", len(getattr(full_geom, "geoms", [full_geom])))


if __name__ == "__main__":
    main()
