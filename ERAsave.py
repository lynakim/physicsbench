import xarray as xr
import pandas as pd
import numpy as np
steps = 40
# Create sample times at 12-hour intervals for all of 2020
start_time = '2020-01-01T00:00:00'
end_time = '2020-12-31T12:00:00'
sample_times = pd.date_range(start=start_time, end=end_time, freq='12H').to_numpy(dtype='datetime64[ns]')

# Load the data
erazarr = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')

# Create a list to store all era_batch datasets
era_batches = []
counter = 0
# Load data for each sample time
for sample_time in sample_times:
    if counter % 10 == 0:
        print('count='+str(counter))
    era_batch = erazarr.sel(time=slice(sample_time + np.timedelta64(6, 'h'), sample_time + np.timedelta64((steps)*6, 'h')))
    era_batches.append(era_batch)
    counter+=1

# Combine all batches into a single dataset with a new dimension
era_batch = xr.concat(era_batches, dim='sample_time')
era_batch['sample_time'] = sample_times

era_batch.to_netcdf("ERA_subsamp_2020.nc")

