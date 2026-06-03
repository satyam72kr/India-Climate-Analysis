#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:22:03 2026

@author: spt
"""

import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy import stats
#%%
path = '/home/spt/25CL06010/data/ERA5_Temp_1978_2025.nc'
data=xr.open_dataset(path)

shape_file_path = '/home/spt/25CL06010/data/India_with_states_shape_file/India_State_Boundary.shp'
states = gpd.read_file(shape_file_path)
states = states.to_crs('EPSG:4326')

#%%
# Kelvin to Celsius
temp = data['t2m'] - 273.15

# Daily maximum
daily_max = temp.resample(valid_time='1D').max()

# Annual mean
annual_mean = daily_max.resample(valid_time='1YE').mean()

# Baseline 1991-2020 (WMO standard)
baseline = annual_mean.sel(valid_time=slice('1991', '2020')).mean('valid_time')

# Anomaly
anomaly = annual_mean - baseline

#%%
# India spatial mean
india_anomaly = anomaly.mean(dim=['latitude', 'longitude'])
years = india_anomaly.valid_time.dt.year.values

# Trend
slope, intercept, r, p, se = stats.linregress(years, india_anomaly.values)
trend_line = slope * years + intercept

print(f"Warming rate: +{slope:.4f}°C per year")
print(f"Total warming 1978-2025: {slope*47:.2f}°C")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(years, india_anomaly.values,
       color=['red' if x > 0 else 'blue' for x in india_anomaly.values],
       alpha=0.6)
ax.plot(years, india_anomaly.values,
        color='black', linewidth=1.5, label='Annual Anomaly')
ax.plot(years, trend_line,
        color='darkred', linewidth=2.5, linestyle='--',
        label=f'Trend: +{slope:.3f}°C/year')
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Temperature Anomaly (°C)', fontsize=12)
ax.set_title('India Mean Temperature Anomaly (1978–2025)\nBaseline: 1991–2020',
             fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('india_anomaly_1978_2025.png', dpi=300)
plt.show()


#%%
# Recent decade anomaly (2015-2024)
recent_anomaly = anomaly.sel(valid_time=slice('2015', '2024')).mean('valid_time')

fig, ax = plt.subplots(figsize=(10, 8),subplot_kw={'projection': ccrs.PlateCarree()})

im = ax.contourf(recent_anomaly.longitude,
                 recent_anomaly.latitude,
                 recent_anomaly.values,
                 levels=20,
                 cmap='RdBu_r',
                 transform=ccrs.PlateCarree())

states.boundary.plot(ax=ax,color='black', linewidth=0.5,transform=ccrs.PlateCarree())


plt.colorbar(im, ax=ax, label='Temperature Anomaly (°C)', shrink=0.6)
ax.set_title('India Temperature Anomaly\n(2015–2024 vs 1991–2020 Baseline)',
             fontsize=14)
plt.savefig('india_spatial_anomaly_updated.png', dpi=300)
plt.show()

#%%
heatwave_days = (daily_max > 40).sum(dim='valid_time')

fig, ax = plt.subplots(figsize=(10, 8),
          subplot_kw={'projection': ccrs.PlateCarree()})

im = ax.contourf(heatwave_days.longitude,
                 heatwave_days.latitude,
                 heatwave_days.values,
                 levels=20,
                 cmap='YlOrRd',
                 transform=ccrs.PlateCarree())

states.boundary.plot(ax=ax,
                     color='black',
                     linewidth=0.5,
                     transform=ccrs.PlateCarree())

plt.colorbar(im, ax=ax, label='Number of Heatwave Days', shrink=0.6)
ax.set_title('India Heatwave Days (1978–2025)\nMax Temp > 40°C', fontsize=14)
plt.savefig('india_heatwave_updated.png', dpi=300)
plt.show()

#%%
# 20 year periods
period1 = annual_mean.sel(
    valid_time=slice('1978', '1997')).mean('valid_time')

period2 = annual_mean.sel(
    valid_time=slice('2006', '2025')).mean('valid_time')

# Difference
diff = period2 - period1

fig, ax = plt.subplots(figsize=(10, 8),
          subplot_kw={'projection': ccrs.PlateCarree()})

im = ax.contourf(diff.longitude,
                 diff.latitude,
                 diff.values,
                 levels=20,
                 cmap='RdBu_r',
                 transform=ccrs.PlateCarree())

states.boundary.plot(ax=ax,
                     color='black',
                     linewidth=0.5,
                     transform=ccrs.PlateCarree())

plt.colorbar(im, ax=ax, 
             label='Temperature Difference (°C)', 
             shrink=0.6)
ax.set_title('India Temperature Change\n(2006–2025 minus 1978–1997)',
             fontsize=14)

plt.savefig('india_20yr_difference.png', dpi=300)
plt.show()


