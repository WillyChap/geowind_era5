import logging
import os

# Suppress gRPC/gcsfs fork warnings that clutter student output
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
import xarray as xr

_ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_STORAGE_OPTIONS = {"token": "anon"}

# Physical constants
_G = 9.80665        # gravitational acceleration, m s-2
_OMEGA = 7.2921e-5  # Earth's rotation rate, rad s-2
_A = 6_371_000.0    # Earth's mean radius, m

# Module-level cache so the zarr store is only opened once per session
_DS_CACHE = None


def _get_dataset() -> xr.Dataset:
    global _DS_CACHE
    if _DS_CACHE is None:
        _DS_CACHE = xr.open_zarr(
            _ZARR_URL,
            storage_options=_STORAGE_OPTIONS,
            consolidated=True,
            chunks={},
        )
    return _DS_CACHE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def open_geopotential(
    time_slice: tuple,
    level: int | list[int] = 500,
    lat: float | tuple | slice | None = None,
    lon: float | tuple | slice | None = None,
) -> xr.DataArray:
    """Open ERA5 geopotential (m2 s -2) at one or more pressure levels.

    Data are loaded lazily — nothing is downloaded until you call
    ``.compute()`` / ``.load()`` or use the :func:`load` helper.

    Parameters
    ----------
    time_slice : (start, end)
        ISO date strings, e.g. ``("2010-01-01", "2010-01-03")``.
        Required — ERA5 is hourly and global; always subset before loading.
    level : int or list[int]
        Pressure level(s) in hPa. Default ``500``.
        Available levels: 1–1000 hPa (37 levels).
    lat : float, (south, north), or slice, or None
        Latitude selection.  Scalar --> nearest grid point.  Tuple or slice -->
        bounding range.  ERA5 latitude runs 90 --> −90; handled internally.
    lon : float, (west, east), or slice, or None
        Longitude. Negative  W values accepted and converted to 0–360.

    Returns
    -------
    xr.DataArray
        Lazy DataArray of geopotential in m2 s -2.
        Divide by ``9.80665`` to get geopotential height in metres.

    Examples
    --------
    ::

        phi = open_geopotential(
            ("2010-01-01", "2010-01-02"),
            level=500,
            lat=(20.0, 60.0),
            lon=(-135.0, -60.0),
        )
        Z = phi / 9.80665  # geopotential height (m)
    """
    ds = _get_dataset()
    da = ds["geopotential"]
    da = da.sel(time=slice(*time_slice))
    da = da.sel(level=level)
    da = _sel_spatial(da, lat, lon)
    # Re-chunk so each Dask task is small: the native zarr chunks bundle all
    # 37 pressure levels together (~147 MB uncompressed per time step).
    # After the level selection above that dimension is gone, but zarr still
    # has to decompress the full chunk.  Splitting spatially here ensures that
    # compute() never holds more than a few hundred MB at once.
    da = da.chunk({"time": 1, "latitude": 181, "longitude": 360})
    return da


def geostrophic_wind(
    phi: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute geostrophic wind components from ERA5 geopotential.

    Uses the quasi-geostrophic approximation:

    .. math::

        u_g = -\\frac{1}{f} \\frac{\\partial \\Phi}{\\partial y}, \\quad
        v_g = +\\frac{1}{f} \\frac{\\partial \\Phi}{\\partial x}

    Spatial gradients are estimated with centred finite differences via
    ``xarray.DataArray.differentiate``.

    Parameters
    ----------
    phi : xr.DataArray
        Geopotential in m2 s -2 with **latitude** and **longitude** dimensions.
        Must cover a spatial region — a single grid point returns NaN.
        Not physically valid within ~5  of the equator (f --> 0).

    Returns
    -------
    ug, vg : xr.DataArray
        Zonal (eastward) and meridional (northward) geostrophic wind in m s-1.

    Examples
    --------
    ::

        phi = open_geopotential(("2010-01-01", "2010-01-01"), level=500,
                                lat=(20.0, 60.0), lon=(-135.0, -60.0))
        phi = load(phi)
        ug, vg = geostrophic_wind(phi)
        ug.isel(time=0).plot()
    """
    lat_rad = np.deg2rad(phi.latitude)
    f = 2.0 * _OMEGA * np.sin(lat_rad)  # Coriolis parameter (s-1)

    # --- meridional gradient ------------------------------------------------
    # differentiate("latitude") --> dphi/dlat in m2 s -2 per degree
    # × (pi/180) converts degrees-->radians; ÷ _A converts arc-->metres
    dPhi_dlat = phi.differentiate("latitude")
    dPhi_dy = dPhi_dlat * (np.pi / 180.0) / _A          # per metre (northward)

    # --- zonal gradient -----------------------------------------------------
    dPhi_dlon = phi.differentiate("longitude")
    dPhi_dx = dPhi_dlon * (np.pi / 180.0) / (_A * np.cos(lat_rad))  # per metre

    ug = -(1.0 / f) * dPhi_dy
    vg = (1.0 / f) * dPhi_dx

    ug.attrs.update({"units": "m s-1", "long_name": "geostrophic zonal wind"})
    vg.attrs.update({"units": "m s-1", "long_name": "geostrophic meridional wind"})

    return ug, vg


def load(da: xr.DataArray, desc: str | None = None) -> xr.DataArray:
    """Download a lazy DataArray and show a progress bar.

    Parameters
    ----------
    da : xr.DataArray
        Lazy DataArray returned by :func:`open_geopotential`.
    desc : str or None
        Label shown next to the progress bar. Defaults to the variable name.

    Returns
    -------
    xr.DataArray
        The same DataArray with data fully loaded into memory.
    """
    from dask.diagnostics import ProgressBar

    label = desc or da.name or "Downloading ERA5"
    print(label)
    with ProgressBar(dt=0.5):
        return da.compute()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sel_spatial(da: xr.DataArray, lat, lon) -> xr.DataArray:
    """Subset on ERA5's regular lat/lon grid.

    ERA5 latitude **decreases** from 90 to −90, so range slices must be
    passed as (max, min).  This is handled automatically.
    """
    if lat is not None:
        if np.isscalar(lat):
            da = da.sel(latitude=float(lat), method="nearest")
        elif isinstance(lat, slice):
            lo = float(lat.start) if lat.start is not None else -90.0
            hi = float(lat.stop) if lat.stop is not None else 90.0
            da = da.sel(latitude=slice(max(lo, hi), min(lo, hi)))
        else:
            lo, hi = float(lat[0]), float(lat[1])
            da = da.sel(latitude=slice(max(lo, hi), min(lo, hi)))

    if lon is not None:
        if np.isscalar(lon):
            da = da.sel(longitude=float(lon) % 360, method="nearest")
        elif isinstance(lon, slice):
            lo_raw = float(lon.start) if lon.start is not None else -180.0
            hi_raw = float(lon.stop) if lon.stop is not None else 180.0
            if abs(hi_raw - lo_raw) < 360.0:
                lo = lo_raw % 360
                hi = hi_raw % 360
                lo, hi = sorted([lo, hi])
                da = da.sel(longitude=slice(lo, hi))
            # else: span is global, skip subsetting
        else:
            lo_raw, hi_raw = float(lon[0]), float(lon[1])
            if abs(hi_raw - lo_raw) < 360.0:
                lo = lo_raw % 360
                hi = hi_raw % 360
                lo, hi = sorted([lo, hi])
                da = da.sel(longitude=slice(lo, hi))
            # else: span is global (-180→180 or 0→360), skip subsetting

    return da
