"""Full-res zoom panels: site aerial + pavement outlines for manual review."""

import os
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(BASE, "data", "raw")
DATA = os.path.join(BASE, "data")

ZOOMS = {
    "z1_west_entrance": (2614150, 343250, 2614700, 343700),
    "z2_tennis_pool": (2614450, 343150, 2615000, 343600),
    "z3_east_upper": (2614850, 343150, 2615400, 343600),
    "z4_ocean_west": (2614250, 342900, 2614800, 343350),
    "z5_ocean_east": (2614750, 342900, 2615300, 343350),
}


def render(name, bbox, outdir):
    with rasterio.open(os.path.join(RAW, "aerial_site_0p25ft.tif")) as src:
        w = from_bounds(*bbox, src.transform)
        img = np.moveaxis(src.read([1, 2, 3], window=w), 0, -1)
        win_bounds = src.window_bounds(w)
    extent = [win_bounds[0], win_bounds[2], win_bounds[1], win_bounds[3]]

    pav = gpd.read_file(os.path.join(DATA, "pavement_ccap_2264.geojson"))
    scope = gpd.read_file(os.path.join(DATA, "parcel_scope_parcels.geojson"))
    bldg = gpd.read_file(os.path.join(DATA, "buildings_2264.geojson"))
    roads = gpd.read_file(os.path.join(RAW, "ncdot_roads_2264.geojson"))
    if roads.crs is None:
        roads = roads.set_crs("EPSG:2264")
    trail = gpd.read_file(os.path.join(RAW, "trails_aoi_2264.geojson"))

    fig, ax = plt.subplots(figsize=(20, 16.4), dpi=110)
    ax.imshow(img, extent=extent)
    pav.boundary.plot(ax=ax, color="red", linewidth=1.4)
    bldg.boundary.plot(ax=ax, color="deepskyblue", linewidth=1.0)
    scope.boundary.plot(ax=ax, color="yellow", linewidth=1.2)
    roads.plot(ax=ax, color="cyan", linewidth=0.8, linestyle="--")
    trail.plot(ax=ax, color="lime", linewidth=0.8, linestyle="--")
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    # 100-ft grid so features can be located in EPSG:2264 coordinates
    for x in range(int(bbox[0]) // 100 * 100, int(bbox[2]) + 100, 100):
        ax.axvline(x, color="white", alpha=0.25, linewidth=0.5)
        ax.text(x, bbox[1] + 4, str(x), color="white", fontsize=6,
                ha="center", alpha=0.8)
    for y in range(int(bbox[1]) // 100 * 100, int(bbox[3]) + 100, 100):
        ax.axhline(y, color="white", alpha=0.25, linewidth=0.5)
        ax.text(bbox[0] + 4, y, str(y), color="white", fontsize=6,
                va="center", alpha=0.8)
    ax.set_axis_off()
    fig.tight_layout(pad=0.1)
    out = os.path.join(outdir, f"{name}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "maps")
    names = sys.argv[2:] or list(ZOOMS)
    for n in names:
        render(n, ZOOMS[n], outdir)
