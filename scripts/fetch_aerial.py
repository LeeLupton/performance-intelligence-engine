"""Export Carteret EagleView 2026 orthoimagery for the Ocean Reef AOI.

ImageServer is native EPSG:2264 with ~0.25 ft pixels; exportImage is
height-capped at 4100 px, so the full-res property export is tiled and
mosaicked. Outputs:
  data/raw/aerial_context_0p5ft.tif  - 0.5 ft/px, AOI + context (review map)
  data/raw/aerial_site_0p25ft.tif    - 0.25 ft/px, property + 150 ft (tracing)
"""

import os

import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from rasterio.transform import from_origin

BASE = ("https://arcgisweb.carteretcountync.gov/arcgis/rest/services/"
        "Imagery/EagleView2026/ImageServer/exportImage")
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

AOI_CONTEXT = (2612599.0, 342327.0, 2615841.0, 344375.0)   # 3242 x 2048 ft
AOI_SITE = (2614122.0, 342777.0, 2615391.0, 343925.0)      # 1269 x 1148 ft
MAX_H = 4000  # stay under the 4100 px cap


def export_tile(bbox, px, path):
    w = int(round((bbox[2] - bbox[0]) / px))
    h = int(round((bbox[3] - bbox[1]) / px))
    r = requests.get(BASE, params=dict(
        bbox=",".join(f"{v:.3f}" for v in bbox), bboxSR=2264, imageSR=2264,
        size=f"{w},{h}", format="tiff", f="image"), timeout=600)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    # exportImage tiffs are not always tagged; stamp georeferencing from bbox
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()
    profile.update(crs="EPSG:2264",
                   transform=from_origin(bbox[0], bbox[3], px, px))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def export_area(bbox, px, out):
    h_px = int(round((bbox[3] - bbox[1]) / px))
    n = int(np.ceil(h_px / MAX_H))
    tiles = []
    for i in range(n):
        y0 = bbox[1] + (bbox[3] - bbox[1]) * i / n
        y1 = bbox[1] + (bbox[3] - bbox[1]) * (i + 1) / n
        tiles.append(export_tile((bbox[0], y0, bbox[2], y1), px,
                                 out + f".tile{i}"))
    srcs = [rasterio.open(t) for t in tiles]
    data, transform = merge(srcs)
    profile = srcs[0].profile.copy()
    profile.update(width=data.shape[2], height=data.shape[1],
                   transform=transform)
    for s in srcs:
        s.close()
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data)
    for t in tiles:
        os.remove(t)
    with rasterio.open(out) as src:
        print(out, src.width, "x", src.height, src.crs,
              "px", src.transform.a, "ft")


if __name__ == "__main__":
    export_area(AOI_CONTEXT, 0.5, os.path.join(RAW, "aerial_context_0p5ft.tif"))
    export_area(AOI_SITE, 0.25, os.path.join(RAW, "aerial_site_0p25ft.tif"))
