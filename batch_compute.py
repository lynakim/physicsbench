import xarray as xr
import numpy as np
import time
import logging
import pandas as pd
from pathlib import Path

def compute_and_store_global_dry_air_mass(input_dataset, output_path):
    """
    Compute global dry air mass from a weather dataset and store the result to a NetCDF file.
    
    This function calculates the global dry air mass using the global_dry_mass function,
    creates a minimal dataset containing only the essential data and metadata,
    and stores it to a compressed NetCDF file.
    
    Args:
        input_dataset (xarray.Dataset): An xarray Dataset containing weather data with 
                                        latitude, longitude, level coordinates and variables
                                        needed for dry air mass calculation (specific_humidity).
        output_path (str): Directory path where the resulting NetCDF file should be saved.
                           If the directory doesn't exist, it will be created.
                           
    Returns:
        tuple: (str, float) - The path to the saved file and the computed global dry air mass value.
        
    Raises:
        ValueError: If required variables are missing from the input dataset.
        IOError: If the output file cannot be written.
    """
    import os
    import time
    import logging
    import xarray as xr
    import pandas as pd
    from pathlib import Path
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('dry_air_mass')
    
    # Check for required coordinates and variables
    required_coords = ['lat', 'lon', 'level']
    required_vars = ['specific_humidity']
    
    for coord in required_coords:
        if coord not in input_dataset.coords:
            raise ValueError(f"Required coordinate '{coord}' not found in the input dataset")
    
    for var in required_vars:
        if var not in input_dataset:
            raise ValueError(f"Required variable '{var}' not found in the input dataset")
    
    # Ensure the output directory exists
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    
    # Get initialization time and forecast lead time from the dataset
    if 'time' in input_dataset.coords:
        init_time = input_dataset.coords['time'].values
    else:
        init_time = None
        logger.warning("No 'time' coordinate found in dataset. Using current time for metadata.")
        init_time = np.datetime64('now')
    
    if 'prediction_timedelta' in input_dataset.coords:
        lead_time = input_dataset.coords['prediction_timedelta'].values
    else:
        lead_time = None
        logger.warning("No 'prediction_timedelta' coordinate found in dataset.")
    
    # Compute global dry air mass
    logger.info("Starting global dry air mass computation...")
    start_time = time.time()
    
    try:
        # Calculate dry air mass using the helper function
        from eval_helpers import global_dry_mass
        dry_mass = global_dry_mass(input_dataset)
        
        # Sum over all dimensions to get the total global value
        global_dry_mass_value = float(dry_mass.sum(dim=['lat', 'lon', 'level']).values)
        
        computation_time = time.time() - start_time
        logger.info(f"Global dry air mass computation completed in {computation_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error calculating global dry air mass: {str(e)}")
        raise
    
    # Create a minimal dataset with just the computed value and essential metadata
    result_ds = xr.Dataset(
        data_vars={
            'global_dry_air_mass': xr.DataArray(
                data=global_dry_mass_value,
                attrs={
                    'long_name': 'Global total dry air mass',
                    'units': 'kg',
                    'description': 'Total mass of dry air in the atmosphere'
                }
            )
        },
        attrs={
            'title': 'Global Dry Air Mass Calculation',
            'history': f'Created on {time.strftime("%Y-%m-%d %H:%M:%S")}',
            'description': 'Global dry air mass calculated from weather model output',
            'computation_time_seconds': computation_time
        }
    )
    
    # Add initialization time and lead time as attributes if available
    if init_time is not None:
        result_ds.attrs['initialization_time'] = str(init_time)
    
    if lead_time is not None:
        result_ds.attrs['forecast_lead_time'] = str(lead_time)
    
    # Generate an informative filename if not specified
    if output_path.endswith('/'):
        init_time_str = 'unknown_init' if init_time is None else pd.to_datetime(init_time).strftime('%Y%m%d_%H%M')
        lead_time_str = 'unknown_lead' if lead_time is None else f"{lead_time}"
        filename = f"global_dry_air_mass_{init_time_str}_{lead_time_str}.nc"
        output_file = os.path.join(output_path, filename)
    else:
        output_file = output_path
    
    # Save the dataset to a compressed NetCDF file
    logger.info(f"Saving global dry air mass to {output_file}")
    try:
        encoding = {'global_dry_air_mass': {'zlib': True, 'complevel': 5}}
        result_ds.to_netcdf(output_file, encoding=encoding)
        logger.info("File saved successfully")
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise IOError(f"Failed to save output file: {str(e)}")
    
    return output_file, global_dry_mass_value

# main function
if __name__ == "__main__":
    # load the dataset
    ds = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')
    # compute the global dry air mass
    output_file, global_dry_mass_value = compute_and_store_global_dry_air_mass(ds, 'gs://weatherbench2/datasets/graphcast_hres_init/2020/global_dry_air_mass')
    print(f"Global dry air mass: {global_dry_mass_value}")