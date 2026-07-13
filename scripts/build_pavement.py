"""Polygonize the C-CAP impervious AOI mask into candidate pavement polygons.

Vectorizes in the raster's native EPSG:5070 (no resampling), reprojects the
vectors to EPSG:2264, subtracts county building footprints, drops slivers,
and lightly simplifies. Output is the ccap-sourced part of the `pavement`
layer; manual traced corrections are added separately.
"""

import os
import sys

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(__file__))
from edge_classify import SIMPLIFY_TOL_FT, SLIVER_MIN_SQFT

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(BASE, "data", "raw")
DATA = os.path.join(BASE, "data")


def main():
    with rasterio.open(os.path.join(RAW, "ccap_impervious_aoi.tif")) as src:
        mask = src.read(1)
        polys = [shape(geom) for geom, val in
                 shapes(mask, mask == 1, transform=src.transform) if val == 1]
    pav = gpd.GeoDataFrame(geometry=polys, crs="EPSG:5070").to_crs("EPSG:2264")

    bldg = gpd.read_file(os.path.join(RAW, "buildings_ocean_reef.geojson"))
    if bldg.crs is None:
        bldg = bldg.set_crs("EPSG:4326")
    bldg = bldg.to_crs("EPSG:2264")
    bmass = unary_union(bldg.geometry.make_valid().values)

    pav["geometry"] = pav.geometry.difference(bmass)
    pav = pav.explode(index_parts=False, ignore_index=True)
    pav = pav[pav.geometry.geom_type == "Polygon"]
    pav = pav[pav.geometry.area >= SLIVER_MIN_SQFT].copy()
    pav["geometry"] = pav.geometry.simplify(SIMPLIFY_TOL_FT,
                                            preserve_topology=True)
    pav["geometry"] = pav.geometry.make_valid()
    pav = pav[~pav.geometry.is_empty].reset_index(drop=True)
    pav["source"] = "ccap"
    pav["feature_type"] = "unknown"

    out = os.path.join(DATA, "pavement_ccap_2264.geojson")
    pav.to_file(out, driver="GeoJSON")
    print(f"{len(pav)} polygons, {pav.geometry.area.sum()/43560:.2f} ac -> {out}")
    bldg.to_file(os.path.join(DATA, "buildings_2264.geojson"), driver="GeoJSON")
    print(f"{len(bldg)} building footprints -> data/buildings_2264.geojson")


if __name__ == "__main__":
    main()
