"""Render inspection overlays: aerial backdrop + C-CAP mask + vectors.

Used for the working inspection view and reused by the final review map.
"""

import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(BASE, "data", "raw")
DATA = os.path.join(BASE, "data")
MAPS = os.path.join(BASE, "maps")


def load_aerial(path):
    with rasterio.open(path) as src:
        img = np.moveaxis(src.read()[:3], 0, -1)
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
    return img, extent


def ccap_on_2264(bounds, px=2.0):
    """Reproject the EPSG:5070 impervious mask onto an EPSG:2264 grid."""
    w = int((bounds[2] - bounds[0]) / px)
    h = int((bounds[3] - bounds[1]) / px)
    dst = np.zeros((h, w), np.uint8)
    dst_transform = rasterio.transform.from_origin(bounds[0], bounds[3], px, px)
    with rasterio.open(os.path.join(RAW, "ccap_impervious_aoi.tif")) as src:
        reproject(rasterio.band(src, 1), dst,
                  dst_transform=dst_transform, dst_crs="EPSG:2264",
                  resampling=Resampling.nearest)
    return dst, [bounds[0], bounds[2], bounds[1], bounds[3]]


def main(out=os.path.join(MAPS, "inspect_overlay.png"), zoom=None):
    img, extent = load_aerial(os.path.join(RAW, "aerial_context_0p5ft.tif"))
    scope = gpd.read_file(os.path.join(DATA, "parcel_scope_parcels.geojson"))
    diss = gpd.read_file(os.path.join(DATA, "parcel_scope_dissolved.geojson"))
    bldg = gpd.read_file(os.path.join(RAW, "buildings_ocean_reef.geojson"))
    roads = gpd.read_file(os.path.join(RAW, "ncdot_roads_2264.geojson"))
    if bldg.crs is None:
        bldg = bldg.set_crs("EPSG:4326")
    bldg = bldg.to_crs("EPSG:2264")
    if roads.crs is None:
        roads = roads.set_crs("EPSG:2264")

    bounds = (extent[0], extent[2], extent[1], extent[3])
    mask, mext = ccap_on_2264(bounds)

    fig, ax = plt.subplots(figsize=(22, 14), dpi=150)
    ax.imshow(img, extent=extent)
    red = np.zeros((*mask.shape, 4), np.float32)
    red[mask == 1] = [1, 0.1, 0.1, 0.45]
    ax.imshow(red, extent=mext)
    roads.plot(ax=ax, color="cyan", linewidth=1.2)
    bldg.boundary.plot(ax=ax, color="deepskyblue", linewidth=1.0)
    scope[scope.scope_class == "core"].boundary.plot(
        ax=ax, color="yellow", linewidth=2.0)
    scope[scope.scope_class == "extended"].boundary.plot(
        ax=ax, color="orange", linewidth=2.0)
    if zoom:
        ax.set_xlim(zoom[0], zoom[2])
        ax.set_ylim(zoom[1], zoom[3])
    else:
        full = diss[diss.scope == "core_plus_extended"].geometry.iloc[0]
        b = full.buffer(250).bounds
        ax.set_xlim(b[0], b[2])
        ax.set_ylim(b[1], b[3])
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 6:
        main(sys.argv[1], zoom=[float(v) for v in sys.argv[2:6]])
    else:
        main()
