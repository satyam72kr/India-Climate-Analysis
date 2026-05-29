#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:06:12 2026

@author: spt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
  India Temperature Anomaly & Heatwave Analysis (1990-2025)
=============================================================
Author      : 25CL06010
Institution : IIT Bhubaneswar
Data        : ERA5 3-hourly Reanalysis | IMD Observational
Description : Analysis of India's warming trends and heatwave
              patterns using ERA5 reanalysis data.
=============================================================
"""
#%%
# 1. IMPORTS

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy import stats

#%%
# 2. LOAD DATA

PATH = '/home/spt/25CL06010/data/netcdf_Temp_19190_2025.nc'
data = xr.open_dataset(PATH)

lat    = data['latitude']   # 38°N to 6°N
lon    = data['longitude']  # 68°E to 98°E
time   = data['valid_time'] # 1990-01-01 to 2025-12-31
temp_k = data['t2m']        # 2m Temperature (Kelvin)

#%%
# 3. PREPROCESSING

# Kelvin → Celsius
temp = temp_k - 273.15

# 3-hourly → Daily Maximum
daily_max = temp.resample(valid_time='1D').max()

# Daily → Annual Mean
annual_mean = daily_max.resample(valid_time='1YE').mean()

#%%

# 4. TEMPERATURE ANOMALY

# Baseline period: 1990–2010
baseline = annual_mean.sel(
    valid_time=slice('1990', '2010')
).mean('valid_time')

# Anomaly = Annual Mean - Baseline
anomaly = annual_mean - baseline

# India-wide spatial mean (one value per year)
india_anomaly = anomaly.mean(dim=['latitude', 'longitude'])
years = india_anomaly.valid_time.dt.year.values

#%%

# 5. STATISTICS

# Max & Min anomaly years
max_idx = india_anomaly.values.argmax()
min_idx = india_anomaly.values.argmin()
print(f"Highest anomaly : {years[max_idx]} — {india_anomaly.values[max_idx]:.4f}°C")
print(f"Lowest  anomaly : {years[min_idx]} — {india_anomaly.values[min_idx]:.4f}°C")

# Linear trend
slope, intercept, r, p, se = stats.linregress(years, india_anomaly.values)
trend_line = slope * years + intercept
print(f"Warming rate    : +{slope:.4f}°C per year")
print(f"Total warming   : +{slope * 35:.2f}°C (1990–2025)")

#%%
# 6. PLOT 1 — TIME SERIES ANOMALY

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(years, india_anomaly.values,
       color=['red' if x > 0 else 'blue' for x in india_anomaly.values],
       alpha=0.6)

ax.plot(years, india_anomaly.values,
        color='black', linewidth=1.5, label='Annual Anomaly')

ax.plot(years, trend_line,
        color='darkred', linewidth=2.5, linestyle='--',
        label=f'Trend: +{slope:.3f}°C/year  |  Total: +{slope*35:.2f}°C')

ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Temperature Anomaly (°C)', fontsize=12)
ax.set_title('India Mean Temperature Anomaly (1990–2025)\nBaseline: 1990–2010',
             fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('india_temperature_anomaly_trend.png', dpi=300)
plt.show()

#%%
# 7. PLOT 2 — SPATIAL ANOMALY MAP

# Recent decade anomaly (2015–2024)
recent_anomaly = anomaly.sel(
    valid_time=slice('2015', '2024')
).mean('valid_time')

fig, ax = plt.subplots(figsize=(10, 8),
                       subplot_kw={'projection': ccrs.PlateCarree()})

im = ax.contourf(recent_anomaly.longitude,
                 recent_anomaly.latitude,
                 recent_anomaly.values,
                 levels=20,
                 cmap='RdBu_r',
                 transform=ccrs.PlateCarree())

ax.add_feature(cfeature.BORDERS,   linewidth=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.STATES,    linewidth=0.5)

plt.colorbar(im, ax=ax, label='Temperature Anomaly (°C)', shrink=0.6)
ax.set_title('India Temperature Anomaly\n(2015–2024 vs 1990–2010 Baseline)',
             fontsize=14)
plt.savefig('india_spatial_anomaly.png', dpi=300)
plt.show()

#%%
# 8. PLOT 3 — HEATWAVE DAYS MAP

# IMD Definition: Max Temp > 40°C
heatwave_days = (daily_max > 40).sum(dim='valid_time')

fig, ax = plt.subplots(figsize=(10, 8),
                       subplot_kw={'projection': ccrs.PlateCarree()})

im = ax.contourf(heatwave_days.longitude,
                 heatwave_days.latitude,
                 heatwave_days.values,
                 levels=20,
                 cmap='YlOrRd',
                 transform=ccrs.PlateCarree())

ax.add_feature(cfeature.BORDERS,   linewidth=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.STATES,    linewidth=0.5)

plt.colorbar(im, ax=ax, label='Number of Heatwave Days', shrink=0.6)
ax.set_title('India Heatwave Days (1990–2025)\nMax Temp > 40°C',
             fontsize=14)
plt.savefig('india_heatwave_days.png', dpi=300)
plt.show()