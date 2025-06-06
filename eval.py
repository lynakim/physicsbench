# %% 
print('hello')
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
#from dask.diagnostics import ProgressBar


# %%


# %%
sample_time = '2020-01-01T12:00:00'
#sample_time1 = '2020-01-01T12:00:00'
#sample_time2 = '2020-03-01T12:00:00'
#sample_time3 = '2020-05-01T12:00:00'
#sample_time4 = '2020-07-01T12:00:00'
#sample_time = [np.datetime64('2020-01-01T12:00:00') + i * np.timedelta64(1*24, 'h') for i in range (0, 5)]
#sample_time2 = [np.datetime64('2020-07-01T12:00:00') + i * np.timedelta64(7*24, 'h') for i in range (0, 5)]

steps = 12

# %%
Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')
gc_batch = Gzarr.sel(time=sample_time).isel(prediction_timedelta=slice(0, steps))
#gc_batch2 = Gzarr.sel(time=sample_time2).isel(prediction_timedelta=slice(0, steps))

# %%
gc_batch_1 = Gzarr.sel(time=sample_time1).isel(prediction_timedelta=slice(0, steps))
gc_batch_2 = Gzarr.sel(time=sample_time2).isel(prediction_timedelta=slice(0, steps))
gc_batch_3 = Gzarr.sel(time=sample_time3).isel(prediction_timedelta=slice(0, steps))
gc_batch_4 = Gzarr.sel(time=sample_time4).isel(prediction_timedelta=slice(0, steps))

# %%
erazarr = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
era_batch = erazarr.sel(time=slice(np.datetime64(sample_time) + np.timedelta64(6, 'h'), np.datetime64(sample_time) + np.timedelta64((steps)*6, 'h')))
#era_batch = erazarr.sel(time=sample_time)
era_batch = era_batch.rename({'latitude': 'lat', 'longitude': 'lon'})

# %%
# create prediction_timedelta dim for era if need plot
era_batch.coords['prediction_timedelta'] = era_batch['time'].diff('time').astype('timedelta64[ns]')

# %%
print(gc_batch['time'] + gc_batch['prediction_timedelta'].isel(prediction_timedelta=0))
print(era_batch['time'].isel(time=0))

# %%
gc_batch = analyze_mass_conservation(gc_batch)
gc_batch['col_residual'] = gc_batch['continuity_error'].sum(dim='level')
gc_tot_residual = gc_batch['abs_continuity_error'].sum(dim=['lat', 'lon', 'level'])

#gc_batch2 = analyze_mass_conservation(gc_batch2)

# %%
era_batch = analyze_mass_conservation(era_batch, is_era=True)
era_batch['col_residual'] = era_batch['continuity_error'].sum(dim='level')
era_tot_residual = era_batch['abs_continuity_error'].sum(dim=['lat', 'lon', 'level'])

# %%
#gc_mass = global_dry_mass(gc_batch).sum(dim=['lat', 'lon', 'level'])
#gc_mass2 = global_dry_mass(gc_batch2).sum(dim=['lat', 'lon', 'level'])
#era_mass = global_dry_mass(era_batch).sum(dim=['lat', 'lon', 'level'])
#gc_batch['col_mass_loss'] = global_dry_mass(gc_batch).sum(dim='level') / global_dry_mass(era_batch.isel(time=0)).sum(dim='level')

# %%
gc_mass_1 = global_dry_mass(gc_batch_1).sum(dim=['lat', 'lon', 'level'])
gc_mass_2 = global_dry_mass(gc_batch_2).sum(dim=['lat', 'lon', 'level'])
gc_mass_3 = global_dry_mass(gc_batch_3).sum(dim=['lat', 'lon', 'level'])
gc_mass_4 = global_dry_mass(gc_batch_4).sum(dim=['lat', 'lon', 'level'])


# %%
print(f'First value of GC Total Mass, New: {gc_total_mass[0]}')
print(f'First value of ERA Total Mass, New: {era_total_mass[0]}')

# %%
# Plot GC and ERA total mass over time
#t = np.linspace(6, 6 * steps, steps)
t = np.linspace(12, 6 * (steps-1), (steps-2))

#plt.plot(t, gc_mass.sum(dim='time')/5, label='GC Jan')
#plt.plot(t, gc_mass2.sum(dim='time')/5, label='GC Jul')
# for i in range(0, 5):
#    plt.plot(t, gc_mass.isel(time=i), label='GC 1/' + str(i+1))

#plt.plot(t, gc_mass.isel(time=0), label='GC 1/1')
#plt.plot(t, gc_mass[1], label='GC 1/2')
#plt.plot(t, gc_mass[2], label='GC 1/3')
#plt.plot(t, gc_mass[3], label='GC 1/4')
#plt.plot(t, gc_mass[4], label='GC 1/5')
plt.plot(t, era_tot_residual.isel(time=slice(1, -1)), label='ERA 1/1 12Z')
plt.plot(t, gc_tot_residual.isel(prediction_timedelta=slice(1, -1)), label='GC 1/1 12Z')

plt.xlabel('lead time (hr)')
#plt.xlabel('time')
plt.ylabel('absolute residual (kg/m³/s)')
plt.xticks(np.arange(0, 6 * (steps + 1), 12))
plt.xticks(rotation=90) 
plt.title('Global Total Continuity Residual')
plt.legend()
plt.show()

# %%
plot_size = 7
plot_example_variable = 'mean_sea_level_pressure'
plot_example_level = 1000
plot_example_max_steps = 10
plot_example_robust = True
input_dataset = gc_batch
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
#residual_diff = gc_batch['col_residual'].isel(prediction_timedelta=4) - gc_batch['col_residual'].isel(prediction_timedelta=3)

gc_batch['residual_diff'] = gc_batch['col_residual'].isel(prediction_timedelta=4) - gc_batch['col_residual'].isel(prediction_timedelta=3)

# %%
img = plt.imshow(residual_diff, cmap='RdBu_r', aspect='auto', interpolation='nearest')

# Add a colorbar
plt.colorbar(label='Value')

# Add labels and title
plt.title('2D Heatmap Visualization (721 x 1440)', fontsize=14)
plt.xlabel('X-axis (1440 points)', fontsize=12)
plt.ylabel('Y-axis (721 points)', fontsize=12)

plt.tight_layout()
plt.show()
# %%
