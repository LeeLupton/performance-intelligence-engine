"""Review map: classed edge candidates + pavement over the 2026 aerial.

This is the mandatory human-cleanup view (brief §6 step 8): a reviewer
visually verifies BLADE vs EXCLUDE segments before the number prices
anything.
"""

import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from rasterio.windows import from_bounds

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(BASE, "data", "raw")
DATA = os.path.join(BASE, "data")
MAPS = os.path.join(BASE, "maps")

COLORS = {
    "BLADE": "#00ff40",
    "EXCLUDE-BUILDING": "#00b7ff",
    "EXCLUDE-DUNE": "#ff2d95",
    "EXCLUDE-NATURAL": "#b76bff",
    "EXCLUDE-BED-INTERIOR": "#c8a24b",
    "EXCLUDE-SCOPE": "#ff9900",
    "EXCLUDE-SEAM": "#ffff00",
}


def render(bbox, out, dpi=170):
    with rasterio.open(os.path.join(RAW, "aerial_context_0p5ft.tif")) as src:
        w = from_bounds(*bbox, src.transform)
        img = np.moveaxis(src.read([1, 2, 3], window=w), 0, -1)
        wb = src.window_bounds(w)
    extent = [wb[0], wb[2], wb[1], wb[3]]

    edges = gpd.read_file(os.path.join(DATA, "edge_candidates_2264.geojson"))
    pav = gpd.read_file(os.path.join(DATA, "pavement_clean_2264.geojson"))
    scope = gpd.read_file(os.path.join(DATA, "parcel_scope_dissolved.geojson"))

    fig, ax = plt.subplots(figsize=(24, 16), dpi=dpi)
    ax.imshow(img, extent=extent, alpha=0.95)
    pav.plot(ax=ax, facecolor="#666666", edgecolor="none", alpha=0.30)
    scope[scope.scope == "core_plus_extended"].boundary.plot(
        ax=ax, color="white", linewidth=1.2, linestyle=":")
    for cls, color in COLORS.items():
        sel = edges[edges["class"] == cls]
        if len(sel):
            lw = 2.6 if cls == "BLADE" else 1.8
            sel.plot(ax=ax, color=color, linewidth=lw)
    handles = [Line2D([0], [0], color=c, lw=3,
                      label=f"{k}  ({edges[edges['class'] == k].length_ft.sum():,.0f} LF)")
               for k, c in COLORS.items() if len(edges[edges["class"] == k])]
    handles.append(Line2D([0], [0], color="white", lw=1.2, linestyle=":",
                          label="scope (core+extended)"))
    ax.legend(handles=handles, loc="lower left", fontsize=11,
              facecolor="black", labelcolor="white", framealpha=0.6)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(
        "Ocean Reef blade-edge takeoff - ESTIMATE, requires visual cleanup "
        "+ on-site verification (aerial: Carteret EagleView 2026; "
        "impervious: NOAA C-CAP 2021)", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    render((2614150, 342850, 2615400, 343850),
           os.path.join(MAPS, "review_map.png"))
    render((2614150, 343350, 2614800, 343780),
           os.path.join(MAPS, "review_map_z_entrance.png"))
    render((2614550, 343200, 2615050, 343760),
           os.path.join(MAPS, "review_map_z_tennis_pool.png"))
    render((2614900, 343150, 2615400, 343650),
           os.path.join(MAPS, "review_map_z_east.png"))
    render((2614200, 342950, 2614950, 343500),
           os.path.join(MAPS, "review_map_z_ocean.png"))
