import xarray as xr
import numpy as np
from eval_helpers import global_dry_mass
import pandas as pd
import time
import os
from tqdm.auto import tqdm

# Load the data
Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')

total_processing_time = 0
first_file_processed = False
estimated_total_storage_mb = 0
year = 2020

# Pre-fetch available times from Gzarr to optimize the loops
available_times_in_garr = Gzarr.time.data

# Iterate over each month of 2020
for month in range(1, 13):
    print(f"\nProcessing data for {year}-{month:02d}...")
    start_time_month_processing = time.time()

    monthly_all_mass_data = []
    monthly_all_time_data = []
    monthly_all_pred_timedelta_data = []
    
    processed_days_in_month = 0

    try:
        num_days_in_month = pd.Timestamp(f'{year}-{month:02d}-01').days_in_month
    except ValueError:
        print(f"Invalid year or month: {year}-{month:02d}. Skipping this month.")
        continue

    # Iterate over each day of the specified month with a progress bar
    for day in tqdm(range(1, num_days_in_month + 1), desc=f"Days in {year}-{month:02d}", unit="day"):
    # for day in tqdm(range(1, 3), desc=f"Days in {year}-{month:02d}", unit="day"):
        current_day_start_dt = np.datetime64(f'{year}-{month:02d}-{day:02d}T00:00:00')
        current_day_end_dt = np.datetime64(f'{year}-{month:02d}-{day:02d}T12:00:00')
        
        ideal_daily_sample_times = pd.date_range(start=current_day_start_dt, end=current_day_end_dt, freq='12H').to_numpy(dtype='datetime64[ns]')
        
        daily_sample_time = [
            t for t in ideal_daily_sample_times
            if t in available_times_in_garr
        ]
        
        if not daily_sample_time:
            print(f"  No data available for {year}-{month:02d}-{day:02d}, skipping day.") # Optional
            continue

        gc_batch = Gzarr.sel(time=daily_sample_time)
        mass_for_day_times = global_dry_mass(gc_batch).sum(dim=['lat', 'lon', 'level'])
        mass_for_day_times_computed = mass_for_day_times.compute()

        monthly_all_mass_data.append(mass_for_day_times_computed)
        monthly_all_time_data.extend(gc_batch.time.data) 

    if not monthly_all_mass_data:
        print(f"No data processed for {year}-{month:02d}. Skipping file generation for this month.")
        end_time_month_processing = time.time()
        month_duration = end_time_month_processing - start_time_month_processing
        print(f"Skipping {year}-{month:02d} (no data) took {month_duration:.2f} seconds.")
        continue
    
    aggregated_mass = xr.concat(monthly_all_mass_data, dim='time')
    aggregated_mass = aggregated_mass.sortby('time')

    result_ds = xr.Dataset({'mass': aggregated_mass})

    output_filename = f'global_dry_mass_{year}_{month:02d}_aggregated_computed.nc'
    
    # --- Debugging prints --- #
    print("--- Inspecting result_ds before saving ---")
    print(result_ds)
    print("------------------------------------------")
    print(result_ds.info())
    print("------------------------------------------")
    # --- End Debugging prints --- #
    
    result_ds.to_netcdf(output_filename)
    print(f"Saved {output_filename}")
    processed_days_in_month = len(np.unique(result_ds.time.dt.day.values))

    end_time_month_processing = time.time()
    month_duration = end_time_month_processing - start_time_month_processing
    total_processing_time += month_duration
    print(f"Processing and saving for {year}-{month:02d} ({processed_days_in_month} days with data) took {month_duration:.2f} seconds.")

    if not first_file_processed:
        try:
            file_size_bytes = os.path.getsize(output_filename)
            file_size_mb = file_size_bytes / (1024 * 1024)
            estimated_total_storage_mb_for_all_months = file_size_mb * 12 
            print(f"Size of {output_filename}: {file_size_mb:.4f} MB.")
            print(f"Estimated total storage for 12 monthly aggregated files: {estimated_total_storage_mb_for_all_months:.4f} MB (based on this first file).")
            first_file_processed = True
            estimated_total_storage_mb = estimated_total_storage_mb_for_all_months
        except OSError as e:
            print(f"Could not get size of {output_filename} to estimate total storage: {e}")

    break

print(f"\nTotal processing time for all months: {total_processing_time:.2f} seconds.")
if first_file_processed:
    print(f"Based on the first processed monthly aggregated file, the estimated total storage is ~{estimated_total_storage_mb:.4f} MB for 12 monthly files.")
else:
    print("Could not estimate total storage as no monthly aggregated files were processed successfully.")



