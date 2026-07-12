"""Manually digitized layers for the Ocean Reef takeoff (EPSG:2264).

Everything here was traced from the Carteret EagleView 2026 orthos
(0.25 ft/px) on 2026-07-12 and is APPROXIMATE — it exists to type and
class the machine-extracted edges, and must be visually verified on the
review map / on site. Coordinates are NC StatePlane ft (EPSG:2264).

  type zones   - assign feature_type to edge segments (priority order)
  dune_zone    - south/oceanfront dune-sand interface (CAMA, never cut)
  natural_zone - unmaintained dune scrub inside/along the upper block
  farside_zone - north of Emerald Dr centerline: never our frontage
  bed_lines    - visible landscape-bed edges (BLADE, source=manual)
"""

import os

import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RAW = os.path.join(DATA, "raw")

# --- feature_type zones, highest priority first -------------------------
TYPE_ZONES = [
    ("tennis_apron", Polygon([(2614640, 343540), (2614950, 343540),
                              (2614950, 343740), (2614640, 343740)])),
    ("pool_apron", Polygon([(2614635, 343410), (2614800, 343410),
                            (2614800, 343560), (2614635, 343560)])),
    ("pool_apron", Polygon([(2614742, 343445), (2614795, 343445),
                            (2614795, 343555), (2614742, 343555)])),
    ("parking", Polygon([(2614290, 343540), (2614680, 343540),
                         (2614680, 343710), (2614290, 343710)])),  # clubhouse
    ("walk_boardwalk", Polygon([(2614725, 343250), (2614780, 343250),
                                (2614780, 343470), (2614725, 343470)])),
]

DUNE_LINE = [
    (2614230, 343180), (2614380, 343230), (2614500, 343250),
    (2614620, 343270), (2614700, 343280), (2614800, 343285),
    (2614900, 343300), (2615000, 343310), (2615080, 343330),
    (2615150, 343390), (2615200, 343460), (2615260, 343520),
]

NATURAL_POLYS = [
    # shrub/sand patch between clubhouse and tennis court
    Polygon([(2614465, 343545), (2614650, 343545),
             (2614650, 343710), (2614465, 343710)]),
    # dune-scrub strip between upper-block parcels and the multi-use path
    Polygon([(2614465, 343660), (2615240, 343700),
             (2615240, 343790), (2614465, 343760)]),
    # shrubs between tennis court and the northeast units
    Polygon([(2614940, 343620), (2615120, 343620),
             (2615120, 343720), (2614940, 343720)]),
]

# planted-bed interiors: pavement edges inside these meet mulch/plantings/
# gravel, not turf (the beds' turf frontier is carried by BED_LINES instead)
BED_INTERIOR_POLYS = [
    # planted band between pool deck and the lawn toward Ocean Dr
    Polygon([(2614640, 343455), (2614640, 343435), (2614700, 343420),
             (2614760, 343430), (2614760, 343455)]),
    # shuffleboard-court garden east of the pool deck (courts + gravel)
    Polygon([(2614742, 343450), (2614792, 343450),
             (2614792, 343552), (2614742, 343552)]),
]

BED_LINES = [
    ("bed", "pool south bed turf frontier (approx, verify on site)",
     LineString([(2614640, 343432), (2614700, 343416), (2614758, 343426)])),
    ("bed", "shuffleboard garden turf frontier east+south (approx)",
     LineString([(2614790, 343548), (2614790, 343455), (2614745, 343450)])),
    ("bed", "clubhouse stepping-stone planting east of parking (approx)",
     LineString([(2614500, 343600), (2614545, 343612), (2614560, 343640)])),
]


def dune_zone():
    """Everything seaward (south) of the traced dune line."""
    pts = DUNE_LINE + [(2615260, 342700), (2614230, 342700)]
    return Polygon(pts)


def farside_zone():
    """North of the Emerald Dr (NC 58) centerline — never our frontage."""
    roads = gpd.read_file(os.path.join(RAW, "ncdot_roads_2264.geojson"))
    if roads.crs is None:
        roads = roads.set_crs("EPSG:2264")
    em = roads[roads["FullName"].str.contains("Emerald", case=False,
                                                na=False)]
    line = unary_union(em.geometry.values)
    merged = line if isinstance(line, LineString) else None
    if merged is None:
        from shapely.ops import linemerge
        merged = linemerge(line)
        if merged.geom_type != "LineString":
            merged = max(merged.geoms, key=lambda g: g.length)
    left = merged.offset_curve(300)
    right = merged.offset_curve(-300)
    cand_l = Polygon(list(merged.coords) + list(left.coords)[::-1])
    cand_r = Polygon(list(merged.coords) + list(right.coords)[::-1])
    # the far side is the one NOT containing the clubhouse
    probe = gpd.points_from_xy([2614450], [343640])[0]
    for cand in (cand_l, cand_r):
        if cand.is_valid and not cand.contains(probe):
            return cand
    return cand_l.buffer(0) if not cand_l.contains(probe) else cand_r.buffer(0)


def road_zones():
    """Type corridors from NCDOT centerlines: Ocean Dr 13 ft half-width
    (local, no NCDOT width attr; ~20-22 ft paved per imagery), Emerald Dr
    15 ft (24 ft SurfaceWidth + margin)."""
    roads = gpd.read_file(os.path.join(RAW, "ncdot_roads_2264.geojson"))
    if roads.crs is None:
        roads = roads.set_crs("EPSG:2264")
    em = roads[roads["FullName"].str.contains("Emerald", case=False, na=False)]
    rest = roads[~roads.index.isin(em.index)]
    zones = []
    if len(em):
        zones.append(("road", unary_union(em.geometry.values).buffer(15)))
    if len(rest):
        zones.append(("road", unary_union(rest.geometry.values).buffer(13)))
    return zones


def row_verge_zone():
    """Internal Ocean Dr right-of-way corridor. The platted ROW between the
    two parcel blocks is ~30-60 ft wide with only ~20 ft of pavement; the
    turf verge inside it is maintained complex grounds, so edges there are
    countable frontage rather than out-of-scope. Non-Emerald centerlines
    buffered 65 ft, capped to the property extent."""
    from shapely.geometry import box
    roads = gpd.read_file(os.path.join(RAW, "ncdot_roads_2264.geojson"))
    if roads.crs is None:
        roads = roads.set_crs("EPSG:2264")
    em = roads["FullName"].str.contains("Emerald", case=False, na=False)
    inner = unary_union(roads[~em].geometry.values).buffer(65)
    return inner.intersection(box(2614260, 342950, 2615255, 343700))


def trail_zone():
    trail = gpd.read_file(os.path.join(RAW, "trails_aoi_2264.geojson"))
    return ("walk_path", unary_union(trail.geometry.values).buffer(9))


def all_type_zones():
    """(feature_type, polygon) list, priority order."""
    return TYPE_ZONES[:4] + [("walk", TYPE_ZONES[4][1]), trail_zone()] \
        + road_zones()


def save_audit_copies():
    rows = [{"kind": "type_zone", "value": n, "geometry": g}
            for n, g in all_type_zones()]
    rows.append({"kind": "dune_zone", "value": "EXCLUDE-DUNE",
                 "geometry": dune_zone()})
    for p in NATURAL_POLYS:
        rows.append({"kind": "natural_zone", "value": "EXCLUDE-NATURAL",
                     "geometry": p})
    for p in BED_INTERIOR_POLYS:
        rows.append({"kind": "bed_interior_zone",
                     "value": "EXCLUDE-BED-INTERIOR", "geometry": p})
    rows.append({"kind": "farside_zone", "value": "EXCLUDE-SCOPE",
                 "geometry": farside_zone()})
    rows.append({"kind": "verge_zone", "value": "ROW-frontage-include",
                 "geometry": row_verge_zone()})
    gpd.GeoDataFrame(rows, crs="EPSG:2264").to_file(
        os.path.join(DATA, "manual_zones_2264.geojson"), driver="GeoJSON")
    beds = gpd.GeoDataFrame(
        [{"feature_type": t, "note": n, "geometry": g}
         for t, n, g in BED_LINES], crs="EPSG:2264")
    beds.to_file(os.path.join(DATA, "manual_bed_lines_2264.geojson"),
                 driver="GeoJSON")
    print("saved manual zones + bed lines;",
          f"bed LF={beds.geometry.length.sum():.0f}")


if __name__ == "__main__":
    save_audit_copies()
