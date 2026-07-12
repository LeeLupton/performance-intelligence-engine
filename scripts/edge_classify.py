"""Core edge extraction + classification for the Ocean Reef blade-edge takeoff.

Operates entirely in EPSG:2264 (NC StatePlane ft). Input pavement polygons may
come from any source (planimetric / ccap / derived / manual); the union of all
pavement is treated as the hard-surface mass, and its outer boundary is the
candidate turf-contact line. Classification per the operator's spec (brief §5):

  BLADE            turf <-> walk/drive/parking/curb/pool-tennis apron (count)
  EXCLUDE-SEAM     pavement <-> pavement contact (interior edges of the union)
  EXCLUDE-BUILDING boundary running along a building footprint
  EXCLUDE-DUNE     boundary inside the dune/sand interface zone (CAMA - never cut)
  EXCLUDE-SCOPE    boundary outside the property scope (kept only in the
                   frontage allowance for road edges; otherwise dropped)
"""

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import linemerge, unary_union

CRS_FT = "EPSG:2264"

SEG_LEN_FT = 3.0          # segmentation step for classification
BUILDING_TOL_FT = 2.5     # edge within this of a building wall -> EXCLUDE-BUILDING
SLIVER_MIN_SQFT = 50.0    # drop pavement slivers below this area
SIMPLIFY_TOL_FT = 0.75    # light simplification (< 1 ft per brief)


@dataclass
class ClassifyInputs:
    pavement: gpd.GeoDataFrame          # polygons; cols: source, feature_type
    scope: Polygon | MultiPolygon       # dissolved parcel scope
    buildings: gpd.GeoDataFrame | None = None
    dune_zone: Polygon | MultiPolygon | None = None
    # road edges within this distance outside scope still count as frontage
    frontage_allowance_ft: float = 30.0
    road_types: tuple = ("road",)
    extras: dict = field(default_factory=dict)


def clean_pavement(pavement: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop slivers, lightly simplify, fix invalid geometry."""
    pav = pavement.copy()
    pav["geometry"] = pav.geometry.make_valid()
    pav = pav.explode(index_parts=False, ignore_index=True)
    pav = pav[pav.geometry.geom_type == "Polygon"]
    pav = pav[pav.geometry.area >= SLIVER_MIN_SQFT].copy()
    pav["geometry"] = pav.geometry.simplify(SIMPLIFY_TOL_FT, preserve_topology=True)
    pav["geometry"] = pav.geometry.make_valid()
    pav = pav[~pav.geometry.is_empty]
    return pav.reset_index(drop=True)


def subtract_buildings(pav: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    if buildings is None or buildings.empty:
        return pav
    bmass = unary_union(buildings.geometry.make_valid().values)
    out = pav.copy()
    out["geometry"] = out.geometry.difference(bmass)
    out = out.explode(index_parts=False, ignore_index=True)
    out = out[out.geometry.geom_type == "Polygon"]
    out = out[out.geometry.area >= SLIVER_MIN_SQFT]
    return out.reset_index(drop=True)


def _to_lines(geom) -> list[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(_to_lines(g))
        return out
    return []


def _chop(line: LineString, step: float) -> list[LineString]:
    """Split a line into pieces no longer than `step` ft."""
    n = max(1, int(np.ceil(line.length / step)))
    pts = [line.interpolate(i / n, normalized=True) for i in range(n + 1)]
    return [
        LineString([pts[i], pts[i + 1]])
        for i in range(n)
        if pts[i].distance(pts[i + 1]) > 1e-6
    ]


def extract_segments(pavement_union) -> gpd.GeoDataFrame:
    """Outer boundary of the pavement mass, chopped into short segments."""
    segs = []
    for line in _to_lines(pavement_union.boundary):
        segs.extend(_chop(line, SEG_LEN_FT))
    return gpd.GeoDataFrame({"geometry": segs}, crs=CRS_FT)


def seam_lines(pav: gpd.GeoDataFrame, pavement_union) -> gpd.GeoDataFrame:
    """Boundaries of individual pavement features interior to the union =
    pavement-pavement seams. Kept in the output explicitly classed EXCLUDE-SEAM
    so the audit trail shows them rather than silently absorbing them."""
    outer = pavement_union.boundary.buffer(0.1)
    pieces = []
    for _, row in pav.iterrows():
        interior = row.geometry.boundary.difference(outer)
        pieces.extend(_to_lines(interior))
    if not pieces:
        return gpd.GeoDataFrame(
            {"geometry": [], "class": [], "source": [], "feature_type": []},
            crs=CRS_FT, geometry="geometry")
    merged = linemerge(unary_union(pieces))
    lines = _to_lines(merged)
    # seams are shared by two features; each contributes a copy -> dedupe via union above
    return gpd.GeoDataFrame(
        {"geometry": lines,
         "class": "EXCLUDE-SEAM",
         "source": "derived",
         "feature_type": "seam"},
        crs=CRS_FT)


def classify(inp: ClassifyInputs) -> dict:
    """Returns dict with keys: pavement (cleaned), edge_candidates, stats."""
    pav = clean_pavement(inp.pavement)
    pav = subtract_buildings(pav, inp.buildings)

    pavement_union = unary_union(pav.geometry.values)
    segs = extract_segments(pavement_union)
    mids = segs.geometry.interpolate(0.5, normalized=True)

    scope = inp.scope
    in_scope = np.array([scope.covers(p) or scope.distance(p) <= 0.5 for p in mids])

    # nearest pavement feature -> source / feature_type attribution
    joined = gpd.sjoin_nearest(
        gpd.GeoDataFrame(geometry=mids, crs=CRS_FT),
        pav[["geometry", "source", "feature_type"]],
        how="left", distance_col="_d")
    joined = joined[~joined.index.duplicated(keep="first")]
    segs["source"] = joined["source"].values
    segs["feature_type"] = joined["feature_type"].values

    is_road = segs["feature_type"].isin(inp.road_types).values

    near_scope = np.array(
        [scope.distance(p) <= inp.frontage_allowance_ft for p in mids])

    if inp.buildings is not None and not inp.buildings.empty:
        bmass = unary_union(inp.buildings.geometry.make_valid().values)
        near_bldg = np.array([bmass.distance(p) <= BUILDING_TOL_FT for p in mids])
    else:
        near_bldg = np.zeros(len(segs), bool)

    if inp.dune_zone is not None:
        in_dune = np.array([inp.dune_zone.covers(p) for p in mids])
    else:
        in_dune = np.zeros(len(segs), bool)

    cls = np.full(len(segs), "BLADE", object)
    cls[~in_scope] = "EXCLUDE-SCOPE"
    cls[~in_scope & is_road & near_scope] = "BLADE"   # road frontage allowance
    cls[near_bldg] = "EXCLUDE-BUILDING"
    cls[in_dune] = "EXCLUDE-DUNE"
    segs["class"] = cls

    # drop far-outside-scope noise entirely; keep near-scope excludes for audit
    keep = in_scope | near_scope
    segs = segs[keep].reset_index(drop=True)

    edges = _merge_segments(segs)
    seams = seam_lines(pav, pavement_union)
    seams = seams[seams.geometry.intersects(scope.buffer(inp.frontage_allowance_ft))]
    edges = pd.concat([edges, seams], ignore_index=True)
    edges = gpd.GeoDataFrame(edges, crs=CRS_FT)
    edges["length_ft"] = edges.geometry.length

    stats = (edges.groupby(["class", "feature_type"])["length_ft"]
             .sum().round(1).to_dict())
    return {"pavement": pav, "edge_candidates": edges, "stats": stats}


def _merge_segments(segs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve contiguous same-class/type/source segments into clean lines."""
    rows = []
    for (c, ft, src), grp in segs.groupby(["class", "feature_type", "source"]):
        merged = linemerge(unary_union(grp.geometry.values))
        for line in _to_lines(merged):
            rows.append({"geometry": line, "class": c,
                         "feature_type": ft, "source": src})
    return gpd.GeoDataFrame(rows, crs=CRS_FT)
