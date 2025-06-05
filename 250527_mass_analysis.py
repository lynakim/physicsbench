import xarray as xr
import numpy as np
from eval_helpers import global_dry_mass
import pandas as pd
import time
import os

# Load the data
Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')

total_processing_time = 0
first_file_processed = False
estimated_total_storage_gb = 0

# Iterate over each month of 2020
for month in range(1, 13):
    print(f"\nProcessing 2020-{month:02d}...")
    start_time_month = time.time()

    # Define the start and end dates for the current month
    start_date = np.datetime64(f'2020-{month:02d}-01T00:00:00')
    # Use pandas to get the end of the month, ensuring correct day (e.g., 29 for Feb in leap year 2020)
    end_date = pd.to_datetime(start_date).to_period('M').end_time.to_datetime64() + np.timedelta64(12, 'h') # include the last 12h step of the month
    

    # Create sample times for the current month at 12-hour intervals
    # Ensure we only select times available in Gzarr
    available_times_in_garr = Gzarr.time.data
    monthly_sample_time = [
        t for t in pd.date_range(start=start_date, end=end_date, freq='12H').to_numpy(dtype='datetime64[ns]')
        if t in available_times_in_garr
    ]
    
    if not monthly_sample_time:
        print(f"No data available for 2020-{month:02d}, skipping.")
        end_time_month = time.time()
        month_duration = end_time_month - start_time_month
        # No processing done, but account for the check time if desired, or just skip total_processing_time update for skipped months.
        # For simplicity, let's not add this small check time to total_processing_time.
        print(f"Skipping 2020-{month:02d} took {month_duration:.2f} seconds (no data).")
        continue

    # monthly_sample_time = [np.datetime64('2020-08-01T12:00:00') + i * np.timedelta64(12, 'h') for i in range (0, 2)]


    gc_batch = Gzarr.sel(time=monthly_sample_time)

    # Calculate global dry mass
    mass = global_dry_mass(gc_batch).sum(dim=['lat', 'lon', 'level'])

    # Create a new dataset with only the required parameters
    result_ds = xr.Dataset({
        'mass': mass,
        'time': gc_batch.time,
        'prediction_timedelta': gc_batch.prediction_timedelta
    })

    # Save to netCDF file, named by month
    output_filename = f'global_dry_mass_2020_{month:02d}.nc'
    result_ds.to_netcdf(output_filename)
    print(f"Saved {output_filename}")

    end_time_month = time.time()
    month_duration = end_time_month - start_time_month
    total_processing_time += month_duration
    print(f"Processing and saving for 2020-{month:02d} took {month_duration:.2f} seconds.")

    if not first_file_processed:
        try:
            file_size_bytes = os.path.getsize(output_filename)
            file_size_mb = file_size_bytes / (1024 * 1024)
            estimated_total_storage_mb = file_size_mb * 12 
            print(f"Size of {output_filename}: {file_size_mb:.4f} MB.")
            print(f"Estimated total storage for 12 months: {estimated_total_storage_mb:.4f} MB (based on this first file). Note: actual total depends on data availability and size for each month.")
            first_file_processed = True
        except OSError as e:
            print(f"Could not get size of {output_filename} to estimate total storage: {e}")
    break

print(f"\nTotal processing time for all months: {total_processing_time:.2f} seconds.")
if first_file_processed:
    print(f"Based on the first processed file, the estimated total storage is ~{estimated_total_storage_mb:.4f} MB for 12 files.")
else:
    print("Could not estimate total storage as no files were processed successfully.")



