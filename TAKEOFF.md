# Ocean Reef Blade-Edge Takeoff — ESTIMATE

**Property:** Ocean Reef condo regime, 2500–2519 Ocean Dr, Emerald Isle NC 28594 (Carteret County)
**Metric:** linear feet of turf-to-hard-surface edge a stick edger blades weekly
**Produced:** 2026-07-12 · all length math in EPSG:2264 (NC StatePlane, US survey ft)

> **THIS NUMBER IS AN ESTIMATE.** It is machine-derived from a 1 m impervious
> raster plus manually digitized zones, and **requires visual cleanup on the
> review map and on-site verification before it prices a bid.** See
> Limitations.

---

## 1. Result

| Measure | LF |
|---|---:|
| **TOTAL BLADE — working scope (core + extended tracts)** | **7,581** |
| BLADE within brief's literal owner-rule scope only (+30 ft frontage) | 4,727 |
| Sensitivity: BLADE lines smoothed 1.5 ft | 7,332 |
| Sensitivity: BLADE lines smoothed 3.0 ft (≈ 1 source pixel) | 6,658 |

**Sanity band check (brief §8: prior visual estimate 5,500–7,000 LF).** The
raw machine total (7,581) is ~8 % above the top of the band. Investigated,
not tuned: (a) 1 m raster stair-stepping inflates diagonal edge length — the
pixel-scale smoothing sensitivity above brings the total to 6,658–7,332,
overlapping the band; (b) the machine count includes small real features a
visual pass tends to skip (≈10 trash-bin pads, clubhouse parking islands).
Expected post-cleanup landing zone: **≈ 6,500–7,300 LF**, consistent with the
band. Do not price from the raw number without the visual cleanup pass.

### Subtotals by feature type (BLADE only, raw)

| Feature group | LF |
|---|---:|
| Road frontage (Ocean Dr internal loop + Emerald Dr near side) | 881 |
| Building pads / drives / parking (incl. clubhouse lot 466) | 4,541 |
| Walks (central beach boardwalk upland portion + multi-use path near entrance) | 593 |
| Pool + tennis aprons | 1,229 |
| Landscape bed lines (manual trace — entrance/pool/clubhouse) | 338 |
| **Total** | **7,581** |

### Excluded edge accounting (kept in `edge_candidates`, classed, not counted)

| Class | LF | Meaning |
|---|---:|---|
| EXCLUDE-BUILDING | 2,540 | edge runs along a building wall/footprint |
| EXCLUDE-NATURAL | 1,636 | edge faces unmaintained dune scrub (path strip, tennis N/W, clubhouse-east patch) |
| EXCLUDE-DUNE | 848 | turf/pavement ↔ dune-sand interface — CAMA-protected, never cut |
| EXCLUDE-BED-INTERIOR | 630 | edges inside planted beds/gravel gardens (pool garden, shuffleboard courts) |
| EXCLUDE-SCOPE | 504 | outside property scope (west public lot side, Emerald Dr far side, entrance apron at NC 58) |
| Pavement↔pavement seams | 0 by construction | edges are extracted from the boundary of the dissolved pavement union, so drive-meets-road and walk-meets-pad seams never enter the candidate set |

## 2. Scope (parcels, dissolved)

Scope was built from named PINs on the Carteret County parcel service and
dissolved in EPSG:2264. Pre-dissolve parcels are layer `parcel_scope`;
dissolved polygons (both variants) are `parcel_scope_dissolved`.

**Core rule (brief §2): 9 parcels, 6.88 ac** — HOA + retained developer
tracts + adjoining Jarrick 2519 tract:

| PIN | Owner | Note | Ac |
|---|---|---|---:|
| 631414436510000 | OCEAN REEF HOMEOWNERS ASSOC | 2500 Ocean Dr, clubhouse/rec parcel | 2.099 |
| 631414433078000 | OCEAN REEF DEVELOPMENT CO | Common area P9 | 0.685 |
| 631414434150000 | OCEAN REEF DEVELOPMENT CO | Common area PH8 | 0.679 |
| 631414435142000 | OCEAN REEF DEVELOPMENT CO | Common area P7 | 0.715 |
| 631414436124000 | OCEAN REEF DEVELOPMENT CO | Common area P6 | 0.715 |
| 631414438577000 | OCEAN REEF DEVELOPMENT CO | Common area PIZ | 0.297 |
| 631414439558000 | OCEAN REEF DEVELOPMENT CO | Common area P11 | 0.297 |
| 631415530264000 | OCEAN REEF DEVELOPMENT CO ETAL | Common area S1 | 0.672 |
| 631414432180000 | JARRICK INC | 2519 Ocean Dr tract; adjoins (0 ft) | 0.716 |

**Extended tracts (working scope adds 10 parcels, 4.79 ac → 19 parcels,
11.66 ac).** A legal-description sweep found the remaining Ocean Reef
common-area land tracts, deeded over time to unit-owner groups (ETAL) or
owners of record rather than retained by the developer. They form the rest
of the complex grounds; excluding them would leave holes in the middle of
the maintained turf. **They fall outside the brief's literal owner rule —
confirm with the client that the maintenance contract covers them:**
631414433447000 (P17), 631414433574000 (P19), 631414434428000 (CONDO),
631414435329000 (P15), 631414437146000 (P5), 631414437596000 (P13),
631414438136000 (P4), 631414439210000 (P3), 631414439273000 (S1 second
tract), 631414437512000 (Jarrick P14).

Also inside the address range but **excluded**: OCEAN IDLERS CONDO at 2516
Ocean Dr (separate association — flag if the contract actually includes it),
66 unit-footprint parcels (unit owners hold no turf), and unrelated Jarrick
parcels in Pier Pointe / Pointe West.

## 3. Data sources and vintages

| Layer | Source | Vintage | Role |
|---|---|---|---|
| Parcels | Carteret Co. `Website/Parcel_Map/MapServer/0` | live service, pulled 2026-07-12 | scope/clip; SITE_HOUSE zero-padding gotcha confirmed and handled |
| Impervious | NOAA C-CAP high-res v2, NC, bulk GeoTIFF (`nc_2021_ccap_v2_hires_impervious.tif`), windowed `/vsicurl/` read — native EPSG:5070 1.0 m pixels, no resampling | **2021** (Ecopia, ≤30 cm stereo imagery + DSM; program window documented 2020–2023) | pavement mask — never final LF |
| Aerial | Carteret `Imagery/EagleView2026/ImageServer`, native EPSG:2264, 0.25 ft px | 2026 | typing, manual zones, review map backdrop |
| Roads | NCDOT RoadNC `RoadNC_RoadData/MapServer` (layers 17/36/29/18/6 joined by RouteID+measure) | attributes dated 2/1/2017, service republished 06/2026 | road context only. **Ocean Dr carries no NCDOT width** (locally maintained); no derived road polygons were used |
| Buildings | Carteret `Layers/County_Building_Footprints/FeatureServer/0` | undated county layer (YEAR_BUILT attrs; complex built 1993) | roofline subtraction + EXCLUDE-BUILDING |
| Multi-use path | Carteret `Layers/Trails/FeatureServer/0` | town layer, GPS/aerial collected | walk typing on Emerald Dr frontage |
| County planimetrics | — | — | **none exist**: all 9 county ArcGIS folders and all 84 open-data Hub items swept; no pavement/sidewalk/edge-of-pavement polygons anywhere, so C-CAP + manual tracing was the required path |

## 4. Method (audit trail)

1. `scripts/build_scope.py` — select 19 PINs, tag core/extended, dissolve (EPSG:2264).
2. `scripts/fetch_ccap_aoi.py` — windowed `/vsicurl/` clip of the NC C-CAP impervious GeoTIFF (bit-identical source pixels, EPSG:5070).
3. `scripts/build_pavement.py` — polygonize mask in 5070 → reproject vectors to 2264 → subtract building footprints → drop slivers < 50 ft² → simplify 0.75 ft (< 1 ft per spec).
4. `scripts/manual_layers.py` — digitized from EagleView 2026: feature-type zones (tennis, pool, clubhouse parking, boardwalk, path, road corridors), dune line, natural-scrub zones, planted-bed interiors, Emerald-far-side exclusion, internal-ROW verge corridor, bed lines. **All approximate.**
5. `scripts/edge_classify.py` + `scripts/run_takeoff.py` — boundary of the dissolved pavement union → 3 ft segments → per-segment class per operator spec §5 → merge contiguous same-class runs → sum BLADE in EPSG:2264.
6. `scripts/render_review_map.py` — mandatory human-cleanup view (`maps/review_map*.png`).

Classification rules implemented exactly per the operator's spec: turf↔walk/
drive/parking/curb/pool/tennis = BLADE; turf↔bed line = BLADE (manual);
pavement↔pavement seams and building edges = EXCLUDE; turf↔dune = EXCLUDE
(CAMA); fences/posts/meters are objects on turf, never edges, and are not in
the candidate set.

## 5. Limitations — read before using the number

- **Visual cleanup is mandatory, not optional.** Every BLADE line on
  `maps/review_map.png` must be confirmed or struck by a human before
  pricing; the machine total WILL move (expected ≈ −5–10 %).
- **C-CAP is a 1 m mask, vintage 2021.** It merges walks/drives/parking into
  blobs, clips corners, stair-steps diagonals (quantified above), and
  predates any 2022–2026 site changes. The EagleView 2026 aerial shows the
  current condition; discrepancies were corrected only where obvious.
- **Manual layers are approximate.** The dune line, natural-scrub zones,
  bed lines, bed interiors and type zones were digitized from imagery at
  ±10–15 ft; the dune interface in particular must be walked on site (CAMA —
  never cut regardless of what any map says).
- **Scope needs client confirmation.** 10 of 19 tracts (4.79 ac) fall outside
  the brief's literal owner rule (see §2). The number for the literal rule
  alone is 4,727 LF.
- **ROW verge assumption.** Edges in the internal Ocean Dr right-of-way strip
  (between the parcel blocks) are counted as maintained frontage; Emerald Dr
  far side and the west public beach-access lot are excluded.
- **Bin pads and small islands** (~10 features) are included as real blade
  edges; strike them on the review map if the crew string-trims them instead.
- **The `pavement` layer keeps context pavement beyond scope** (for review);
  classification clipped edges, not polygons.
- **Nothing here is survey-grade.** Parcel lines are county GIS, not a
  survey; all coordinates are estimate-grade only.

## 6. Deliverables

- `ocean_reef_takeoff.gpkg` — layers: `parcel_scope`,
  `parcel_scope_dissolved`, `road_context`, `pavement`, `buildings`,
  `edge_candidates` (every segment classed + source-tagged: ccap / manual /
  derived), `manual_zones`, `manual_bed_lines`
- `data/*.geojson` — GeoJSON working copies · `data/raw/` — raw service pulls
- `maps/review_map.png` (+ 4 zoom panels) — review/cleanup view
- `scripts/` — full reproducible pipeline
