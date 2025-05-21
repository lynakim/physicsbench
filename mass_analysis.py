import xarray as xr
import numpy as np
from eval_helpers import global_dry_mass

# Load the data
Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')
sample_time = [np.datetime64('2020-01-27T12:00:00') + i * np.timedelta64(12, 'h') for i in range(0, 4)]
gc_batch = Gzarr.sel(time=sample_time).isel(prediction_timedelta=slice(0, 12))

# Calculate global dry mass
mass = global_dry_mass(gc_batch).sum(dim=['lat', 'lon', 'level'])

# Create a new dataset with only the required parameters
result_ds = xr.Dataset({
    'mass': mass,
    'time': gc_batch.time,
    'predictiontime_delta': gc_batch.prediction_timedelta
})

# Save to netCDF file
result_ds.to_netcdf('global_dry_mass.nc') 

