# %%
import dataclasses
import datetime
import functools
import math
import re
from typing import Optional

import cartopy.crs as ccrs
from google.cloud import storage
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray as xr
from IPython.display import HTML
from eval_helpers import *
from plot_helpers import *

# %%
Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')
gc_batch = Gzarr.sel(time='2020-01-27T12:00:00').isel(prediction_timedelta=slice(0, 12))

"""
def prep_gc_format(gc_ds):
    #create prediction_timedelta dim for gc if need plot
    gc_ds['time'] = gc_ds['prediction_timedelta'].diff('time').astype('timedelta64[ns]')
    
    return gc_ds

gc_batch_subset = prep_gc_format(gc_batch_subset)
"""
# %%
erazarr = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
era_batch = erazarr.sel(time=slice('2020-01-27T12:00:00', '2020-02-06T12:00:00')).isel(time=slice(1, 13))

def prep_era_format(era_ds):
    era_ds = era_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    #era_ds = era_ds.isel(time=slice(1, None))

    #create prediction_timedelta dim for era if need plot
    #era_ds.coords['prediction_timedelta'] = era_ds['time'].diff('time').astype('timedelta64[ns]')
    
    return era_ds

era_batch = prep_era_format(era_batch)

# %%
print(gc_batch['time'] + gc_batch['prediction_timedelta'].isel(prediction_timedelta=0))
print(era_batch['time'].isel(time=0))

# %%
gc_batch = analyze_mass_conservation(gc_batch)

# %%
era_batch = analyze_mass_conservation(era_batch, is_era=True)

# %%
# Calulate total mass over time
gc_total_mass = gc_batch['dry_air_mass'].sum(dim=['lat', 'lon', 'level'])

# %%
era_total_mass = era_batch['dry_air_mass'].sum(dim=['lat', 'lon', 'level'])

# %%
# Calulate total mass over time
gc_1000_mass = gc_batch_subset['dry_air_mass'].isel(level=-1).sum(dim=['lat', 'lon'])
era_1000_mass = era_batch_subset['dry_air_mass'].isel(level=-1).sum(dim=['lat', 'lon'])

# %%
#era_total_mass_2 = (era_batch_subset['air_density'] * era_batch_subset['volume']).sum(dim=['lat', 'lon', 'level'])

# %%
print(f'First value of GC Total Mass: {gc_total_mass.values[0]}')
print(f'First value of ERA Total Mass: {era_total_mass.values[0]}')
#print(f'Second value of ERA Total Mass, Old: {era_total_old_mass[0]}')

# %%
t = gc_total_mass.coords['prediction_timedelta'] #/ (3.6e12 * 24)
#t = era_total_old_mass.coords['time']

plt.plot(t, gc_total_mass, label='GC')
#plt.plot(t, gc_total_old_mass, label='GC 5/8')
#plt.plot(t, gc_total_mass - gc_total_old_mass, label='GC diff.')

#plt.plot(t, era_1000_mass, label='ERA')
plt.plot(t, era_total_mass, label='ERA')
#plt.plot(t, era_total_old_mass, label='ERA 5/8')

plt.xlabel('lead time (ns)')
#plt.xlabel('time')
plt.ylabel('total mass (kg)')
plt.xticks(rotation=90)
plt.title('Global Dry Air Mass')
plt.legend()
plt.show()

# %%
t = era_total_mass.coords['time']
plt.plot(t, 100 * (gc_total_mass - era_total_mass, label='ERA diff.')

#plt.plot(t, 100 * (era_total_mass_2 - era_total_mass) / era_total_mass, label='ERA diff.')


plt.xlabel('time')
plt.ylabel('total mass difference (%)')
plt.title('Global Dry Air Mass Percentage Difference Between Two Methods')
plt.xticks(rotation=90)
plt.legend()
plt.show()
# %%
gc_old_density = old_calculate_air_density(gc_batch_subset)
gc_new_density = calculate_air_density(gc_batch_subset)

# %%
density_diff_perc = 100 * (gc_old_density[12][0] - gc_new_density[12][0]) / gc_new_density[12][0]

# %%
# Plot the heatmap
plt.imshow(density_diff_perc, cmap='viridis')

# Add a colorbar for reference
plt.colorbar()

# Customize the plot (optional)
plt.title('Percentage diff. between air densities (kg/m^3), 1000 hPa')
plt.xlabel('longitude')
plt.ylabel('latitude')

# Show the plot
plt.show()

# %%

plot_size = 7
plot_example_variable = 'good_dry_mass'
plot_example_level = 500
plot_example_max_steps = 10
plot_example_robust = True
input_dataset = gc_batch_subset
is_era = False

data = {
    plot_example_variable: scale(
        select(input_dataset, plot_example_variable, plot_example_level, plot_example_max_steps, is_era),
        robust=plot_example_robust
    ),
}

fig_title = plot_example_variable
if "level" in input_dataset[plot_example_variable].coords:
    fig_title += f" at {plot_example_level} hPa"

plot_data(data, fig_title, plot_size, plot_example_robust, is_era=is_era)

# %%
