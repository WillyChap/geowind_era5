# geowind-era5

A lightweight Python package for computing **geostrophic wind** from [ERA5 reanalysis](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) geopotential data via the [ARCO-ERA5](https://github.com/google-research/arco-era5) dataset on Google Cloud Storage. No account or credentials required.

Data are opened **lazily** via Zarr and xarray, so nothing is downloaded until you actually need it.

> **Data source:** `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`  
> Maintained by Google Research. Coverage: 1940–present, 0.25  hourly.

---

## Background

The **geostrophic wind** is the theoretical wind that results from a balance between the horizontal pressure gradient force and the Coriolis force. On an isobaric (constant-pressure) surface it is:

```
u_g = -(1/f) dphi/dy
v_g = +(1/f) dphi/dx
```

where `phi` is the geopotential (m 2 s⁻ 2) and `f = 2omega sin(phi)` is the Coriolis parameter. Geostrophic wind is a good approximation to the actual wind outside of the tropics and away from strong curvature (jet streaks, etc.).

---

## Installation

```bash
pip install -e .
```

For plotting support:

```bash
pip install -e ".[plot]"
```

Requires Python ≥ 3.10.

---

## Quick start

```python
from geowind_era5 import geostrophic_wind, load, open_geopotential

# 1. Open ERA5 geopotential lazily at 500 hPa over the CONUS
phi = open_geopotential(
    time_slice=("2010-01-01", "2010-01-01"),
    level=500,
    lat=(20.0, 60.0),
    lon=(-135.0, -60.0),   # negative  W fine; converted to 0–360 internally
)

# 2. Download (progress bar shown automatically)
phi = load(phi)

# 3. Compute geostrophic wind components
ug, vg = geostrophic_wind(phi)

# 4. Plot zonal component at first time step
ug.isel(time=0).plot()
```

---

## API

### `open_geopotential`

```python
open_geopotential(
    time_slice,           # (start, end) ISO date strings — required
    level=500,            # pressure level(s) in hPa
    lat=None,             # scalar, (south, north), or slice
    lon=None,             # scalar, (west, east), or slice;  W accepted
)  --> xr.DataArray         # lazy, units: m 2 s⁻ 2
```

Divide the result by `9.80665` to get **geopotential height** in metres.

### `geostrophic_wind`

```python
geostrophic_wind(phi)  --> (ug, vg)
```

Takes the geopotential DataArray (loaded or lazy) and returns zonal `ug` and meridional `vg` wind components in m s-1. Requires at least a 2-D spatial domain — not valid at the equator (f  --> 0).

### `load`

```python
load(da, desc=None)  --> xr.DataArray
```

Downloads a lazy DataArray and shows a `dask` progress bar. Equivalent to `da.compute()` but friendlier for students.

---

## Spatial selection

| Form | Behaviour |
|------|-----------|
| `scalar` (e.g. `40.0`) | Nearest grid point |
| `tuple` (e.g. `(37.0, 41.0)`) | Bounding range |
| `slice` (e.g. `slice(37.0, 41.0)`) | Bounding range |
| `None` | No spatial subsetting |

ERA5 latitude **decreases** from 90  --> −90. Pass values in any order — the package handles the flip internally.

---

## Pressure levels

ERA5 has 37 pressure levels (hPa):

> 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, **500**, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000

---

## Grid

| Property | Value |
|----------|-------|
| Spatial resolution | 0.25  (~28 km) |
| Latitude range | 90  --> −90 (decreasing) |
| Longitude range | 0  --> 359.75 |
| Time resolution | Hourly |
| Coverage | 1940–present |

---

## Example

```bash
python examples/geowind_500hPa.py
```

Downloads one time step of 500 hPa geopotential over the CONUS, computes geostrophic wind, and saves a three-panel plot (`geowind_500hPa.png`).
