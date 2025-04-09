
def calculate_column_mass_flux_divergence(ds, is_era=False):
    """
    Calculate the total mass flux divergence by vertical column.
    
    This function:
    1. Multiplies mass_flux_divergence (kg/m³/s) by cell volume (m³) to get mass flux in kg/s
    2. Sums these values over the level dimension to get column-integrated values
    
    Args:
        ds (xarray.Dataset): Dataset containing mass_flux_divergence and volume
        is_era (bool): Whether the dataset is ERA5 format
        
    Returns:
        xarray.DataArray: Column-integrated mass flux divergence in kg/s
    """
    # Multiply mass flux divergence (kg/m³/s) by volume (m³) to get kg/s
    volume_scaled_divergence = ds['mass_flux_divergence'] * ds['volume']
    
    # Sum over the vertical levels to get the column total
    column_divergence = volume_scaled_divergence.sum(dim='level')
    
    return column_divergence

def calculate_column_mass_tendency(ds, is_era=False):
    """
    Calculate the total mass tendency by vertical column.
    
    This function:
    1. Multiplies air_density_tendency (kg/m³/s) by cell volume (m³) to get mass tendency in kg/s
    2. Sums these values over the level dimension to get column-integrated values
    
    Args:
        ds (xarray.Dataset): Dataset containing air_density_tendency and volume
        is_era (bool): Whether the dataset is ERA5 format
        
    Returns:
        xarray.DataArray: Column-integrated mass tendency in kg/s
    """
    # Multiply air density tendency (kg/m³/s) by volume (m³) to get kg/s
    volume_scaled_tendency = ds['air_density_tendency'] * ds['volume']
    
    # Sum over the vertical levels to get the column total
    column_tendency = volume_scaled_tendency.sum(dim='level')
    
    return column_tendency

def analyze_column_mass_conservation(ds, is_era=False):
    """
    Perform column-wise mass conservation analysis on a dataset.
    
    Args:
        ds (xarray.Dataset): Input dataset with mass_flux_divergence and air_density_tendency
        is_era (bool): Whether the dataset is ERA5
        
    Returns:
        xarray.Dataset: Dataset with added column mass conservation variables
    """
    # Calculate column-integrated values
    ds['column_mass_flux_divergence'] = calculate_column_mass_flux_divergence(ds, is_era)
    ds['column_mass_tendency'] = calculate_column_mass_tendency(ds, is_era)
    
    # Calculate column continuity error
    ds['column_continuity_error'] = abs(ds['column_mass_tendency'] - ds['column_mass_flux_divergence'])
    
    # Calculate relative error (as percentage of total column mass change)
    # Adding a small epsilon to avoid division by zero
    epsilon = 1e-10
    ds['column_relative_error'] = (ds['column_continuity_error'] / 
                                  (abs(ds['column_mass_tendency']) + epsilon)) * 100
    
    return ds

# Example usage:
# gc_with_column_analysis = analyze_column_mass_conservation(gc_batch_subset)
# era_with_column_analysis = analyze_column_mass_conservation(era_batch_subset, is_era=True)

# To visualize the results:
# gc_with_column_analysis['column_mass_flux_divergence'].isel(prediction_timedelta=1).plot()
# gc_with_column_analysis['column_mass_tendency'].isel(prediction_timedelta=1).plot()
# gc_with_column_analysis['column_continuity_error'].isel(prediction_timedelta=1).plot()


# Example usage:
gc_with_column_analysis = analyze_column_mass_conservation(gc_batch_subset)
# era_with_column_analysis = analyze_column_mass_conservation(era_batch_subset, is_era=True)

# To visualize the results:
gc_with_column_analysis['column_mass_flux_divergence'].isel(prediction_timedelta=1).plot()
gc_with_column_analysis['column_mass_tendency'].isel(prediction_timedelta=1).plot()
gc_with_column_analysis['column_continuity_error'].isel(prediction_timedelta=1).plot()
# %%
