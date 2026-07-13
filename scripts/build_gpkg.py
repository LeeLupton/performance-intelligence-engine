"""Assemble ocean_reef_takeoff.gpkg from the pipeline outputs.

Layers (all EPSG:2264 unless noted):
  parcel_scope           - 19 selected parcels, scope_class core/extended
  parcel_scope_dissolved - dissolved core_rule / core_plus_extended polygons
  road_context           - NCDOT RoadNC centerlines + joined attributes
  pavement               - cleaned pavement polygons (source-tagged)
  buildings              - county building footprints
  edge_candidates        - classed edge lines (class / feature_type /
                           source / length_ft)
  manual_zones           - digitized type/exclusion zones (audit)
  manual_bed_lines       - digitized bed lines (audit copy; also present
                           in edge_candidates as BLADE/bed/manual)
"""

import os

import geopandas as gpd

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data")
RAW = os.path.join(DATA, "raw")
GPKG = os.path.join(BASE, "ocean_reef_takeoff.gpkg")


def add(path, layer, crs=None):
    g = gpd.read_file(path)
    if g.crs is None and crs:
        g = g.set_crs(crs)
    if str(g.crs) != "EPSG:2264":
        g = g.to_crs("EPSG:2264")
    g.to_file(GPKG, layer=layer, driver="GPKG")
    print(f"{layer}: {len(g)} features")


def main():
    if os.path.exists(GPKG):
        os.remove(GPKG)
    add(os.path.join(DATA, "parcel_scope_parcels.geojson"), "parcel_scope")
    add(os.path.join(DATA, "parcel_scope_dissolved.geojson"),
        "parcel_scope_dissolved")
    add(os.path.join(RAW, "ncdot_roads_2264.geojson"), "road_context",
        crs="EPSG:2264")
    add(os.path.join(DATA, "pavement_clean_2264.geojson"), "pavement")
    add(os.path.join(DATA, "buildings_2264.geojson"), "buildings")
    add(os.path.join(DATA, "edge_candidates_2264.geojson"), "edge_candidates")
    add(os.path.join(DATA, "manual_zones_2264.geojson"), "manual_zones")
    add(os.path.join(DATA, "manual_bed_lines_2264.geojson"),
        "manual_bed_lines")
    print("wrote", GPKG)


if __name__ == "__main__":
    main()
