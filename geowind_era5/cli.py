"""
Command-line interface for geowind-era5.

Usage examples
--------------
# Download 500 hPa geostrophic wind over CONUS and save to NetCDF
geowind --level 500 --lat 20 60 --lon -135 -60 --time 2010-01-01 2010-01-01

# Custom output filename
geowind --level 250 --lat 30 70 --lon -180 -60 --time 2020-02-01 2020-02-03 -o jetstream.nc
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="geowind",
        description="Compute ERA5 geostrophic wind and save to NetCDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--level", type=int, default=500,
        help="Pressure level in hPa (default: 500)",
    )
    parser.add_argument(
        "--lat", nargs=2, type=float, metavar=("SOUTH", "NORTH"),
        default=None,
        help="Latitude range, e.g. --lat 20 60",
    )
    parser.add_argument(
        "--lon", nargs=2, type=float, metavar=("WEST", "EAST"),
        default=None,
        help="Longitude range, e.g. --lon -135 -60  (negative °W fine)",
    )
    parser.add_argument(
        "--time", nargs=2, metavar=("START", "END"),
        required=True,
        help="Time range as ISO date strings, e.g. --time 2010-01-01 2010-01-01",
    )
    parser.add_argument(
        "--output", "-o", default="geowind.nc",
        help="Output NetCDF filename (default: geowind.nc)",
    )

    args = parser.parse_args()

    # Import here so startup is fast when --help is called
    import xarray as xr
    from geowind_era5 import geostrophic_wind, load, open_geopotential

    lat = tuple(args.lat) if args.lat else None
    lon = tuple(args.lon) if args.lon else None

    print(f"Opening ERA5 geopotential — {args.level} hPa  {args.time[0]} → {args.time[1]}")
    phi = open_geopotential(
        time_slice=tuple(args.time),
        level=args.level,
        lat=lat,
        lon=lon,
    )

    phi = load(phi, desc=f"{args.level} hPa geopotential")

    print("Computing geostrophic wind...")
    ug, vg = geostrophic_wind(phi)

    ds = xr.Dataset(
        {"geopotential": phi, "ug": ug, "vg": vg},
        attrs={"level_hPa": args.level, "source": "ARCO-ERA5"},
    )
    ds.to_netcdf(args.output)
    print(f"Saved → {args.output}")
