"""Clip NOAA C-CAP NC 2021 high-res impervious to the Ocean Reef AOI.

Windowed /vsicurl/ read of the bulk statewide GeoTIFF — native EPSG:5070
1.0 m source pixels, no reprojection/resampling. Source vintage: 2021
(C-CAP Version 2, Ecopia, <=30 cm stereo imagery + DSM; program window
documented 2020-2023).
"""

import os

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

SRC = ("/vsicurl/https://ocmgeodatastor1.blob.core.windows.net/ccap/"
       "bulk_download/C-CAP_High-Resolution_Data/"
       "Initial_C-CAP_High-Resolution_Land_Cover_Layers/Impervious/CONUS/"
       "nc_2021_ccap_v2_hires_impervious.tif")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                   "ccap_impervious_aoi.tif")

# In-scope parcel bbox (EPSG:2264) + 600 ft context so bordering road
# pavement (Ocean Dr / Emerald Dr) is captured.
AOI_2264 = (2612599.0, 342327.0, 2615841.0, 344375.0)


def main():
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    tr = Transformer.from_crs("EPSG:2264", "EPSG:5070", always_xy=True)
    xs = np.linspace(AOI_2264[0], AOI_2264[2], 50)
    ys = np.linspace(AOI_2264[1], AOI_2264[3], 50)
    edge = ([(x, AOI_2264[1]) for x in xs] + [(x, AOI_2264[3]) for x in xs]
            + [(AOI_2264[0], y) for y in ys] + [(AOI_2264[2], y) for y in ys])
    tx, ty = tr.transform(*zip(*edge))
    xmin, ymin, xmax, ymax = min(tx), min(ty), max(tx), max(ty)

    with rasterio.open(SRC) as src:
        row0, col0 = src.index(xmin, ymax)
        row1, col1 = src.index(xmax, ymin)
        w = Window(col0, row0, col1 - col0 + 1, row1 - row0 + 1)
        data = src.read(1, window=w)
        profile = src.profile.copy()
        profile.update(width=int(w.width), height=int(w.height),
                       transform=src.window_transform(w))
        with rasterio.open(OUT, "w", **profile) as dst:
            dst.write(data, 1)

    vals, cnt = np.unique(data, return_counts=True)
    print("source:", SRC.replace("/vsicurl/", ""))
    print("window:", w, "crs EPSG:5070, 1.0 m pixels")
    print("bounds 5070:", (xmin, ymin, xmax, ymax))
    print("value counts:", dict(zip(vals.tolist(), cnt.tolist())))


if __name__ == "__main__":
    main()
