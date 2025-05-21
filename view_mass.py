# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Read the netCDF file
ds = xr.open_dataset('global_dry_mass.nc')

# Prepare axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Get hours for x-axis
hours = ds.predictiontime_delta.values.astype('timedelta64[h]').astype(float)

# If mass is 2D (time, prediction_timedelta), plot each time as a separate line
mass = ds.mass.values
if mass.ndim == 2:
    for i in range(mass.shape[0]):
        ax1.plot(hours, mass[i], label=f"time {i+1}", alpha=0.7)
    # Plot mean line
    mean_mass = np.mean(mass, axis=0)
    ax1.plot(hours, mean_mass, 'k-', label='Mean', linewidth=2)
    # Plot deviation for each time
    for i in range(mass.shape[0]):
        initial_mass = mass[i, 0]
        mass_deviation = (mass[i] - initial_mass) / initial_mass * 100
        ax2.plot(hours, mass_deviation, alpha=0.7)
    # Plot mean deviation
    mean_initial = np.mean(mass[:, 0])
    mean_deviation = (mean_mass - mean_initial) / mean_initial * 100
    ax2.plot(hours, mean_deviation, 'k-', label='Mean', linewidth=2)
else:
    ax1.plot(hours, mass, 'b.-', linewidth=2, markersize=10)
    initial_mass = mass[0]
    mass_deviation = (mass - initial_mass) / initial_mass * 100
    ax2.plot(hours, mass_deviation, 'r.-', linewidth=2, markersize=10)

ax1.set_xlabel('Lead Time (hours)')
ax1.set_ylabel('Global Dry Mass (kg)')
ax1.set_title('Global Dry Mass Over Time')
ax1.grid(True)
ax1.legend()

ax2.set_xlabel('Lead Time (hours)')
ax2.set_ylabel('Mass Deviation (%)')
ax2.set_title('Mass Conservation: Deviation from Initial Mass')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('mass_analysis_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Print basic information about the dataset
print("\nDataset Info:")
print(ds.info())

# Print the actual values
print("\nMass values:")
print(ds.mass.values)

print("\nTime values:")
print(ds.time.values)

print("\nPrediction time delta values:")
print(ds.predictiontime_delta.values) 
# %%
