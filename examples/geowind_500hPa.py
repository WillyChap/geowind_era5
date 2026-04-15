"""
Calculate and plot 500 hPa geostrophic wind over the contiguous US
for 00 UTC 1 January 2010.

Run:
    python examples/geowind_500hPa.py
"""

import matplotlib.pyplot as plt
import numpy as np

from geowind_era5 import geostrophic_wind, load, open_geopotential

# 1. Open geopotential lazily — no data downloaded yet
phi = open_geopotential(
    time_slice=("2010-01-01", "2010-01-01"),
    level=500,
    lat=(20.0, 60.0),
    lon=(-135.0, -60.0),   # negative  W fine; converted to 0–360 internally
)

print("Lazy geopotential DataArray:")
print(phi, "\n")

# 2. Download (progress bar shows download progress)
phi = load(phi, desc="500 hPa geopotential")

# 3. Compute geostrophic wind — in-memory, fast after download
ug, vg = geostrophic_wind(phi)

# 4. First time step only for plotting
phi0 = phi.isel(time=0)
ug0 = ug.isel(time=0)
vg0 = vg.isel(time=0)

print(f"Geostrophic wind range:")
print(f"  u_g: {float(ug0.min()):.1f} to {float(ug0.max()):.1f} m/s")
print(f"  v_g: {float(vg0.min()):.1f} to {float(vg0.max()):.1f} m/s")

# 5. Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Geopotential height (Z =  phi/g)
Z = phi0 / 9.80665
Z.plot(ax=axes[0], cmap="viridis", cbar_kwargs={"label": "Z (m)"})
axes[0].set_title("500 hPa Geopotential Height")

# Zonal geostrophic wind
ug0.plot(ax=axes[1], cmap="RdBu_r", vmin=-60, vmax=60,
         cbar_kwargs={"label": "m s-1"})
axes[1].set_title("500 hPa Geostrophic $u_g$")

# Meridional geostrophic wind
vg0.plot(ax=axes[2], cmap="RdBu_r", vmin=-30, vmax=30,
         cbar_kwargs={"label": "m s-1"})
axes[2].set_title("500 hPa Geostrophic $v_g$")

# Overlay wind barbs (subsampled every 4th point for readability)
lat = phi0.latitude.values
lon = phi0.longitude.values - 360  # shift to negative  W for labelling
LON, LAT = np.meshgrid(lon, lat)
step = 4

for ax in axes:
    ax.barbs(
        LON[::step, ::step], LAT[::step, ::step],
        ug0.values[::step, ::step], vg0.values[::step, ::step],
        length=5, linewidth=0.4, color="k",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.suptitle("ERA5  |  500 hPa Geostrophic Wind  |  2010-01-01 00 UTC",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("geowind_500hPa.png", dpi=150, bbox_inches="tight")
print("Saved geowind_500hPa.png")
plt.show()
