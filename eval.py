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
from dask.diagnostics import ProgressBar


# %%
# create prediction_timedelta dim for era if need plot
# era_batch_subset.coords['prediction_timedelta'] = era_batch_subset['time'].diff('time').astype('timedelta64[ns]')

# %%

plot_size = 7
plot_example_variable = 'continuity_error'
plot_example_level = 1000
plot_example_max_steps = 5
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
sample_time = '2020-08-27T12:00:00'
steps = 10

# %%
Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')
gc_batch = Gzarr.sel(time=sample_time).isel(prediction_timedelta=slice(0, steps))
# %%
erazarr = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
era_batch = erazarr.sel(time=slice(np.datetime64(sample_time) + np.timedelta64(6, 'h'), np.datetime64(sample_time) + np.timedelta64(steps*6, 'h')))
era_batch = era_batch.rename({'latitude': 'lat', 'longitude': 'lon'})

# %%
gc_batch = analyze_mass_conservation(gc_batch)
# %%
era_batch = analyze_mass_conservation(era_batch, is_era=True)