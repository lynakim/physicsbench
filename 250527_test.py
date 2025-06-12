# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %%
# load the data
ds = xr.open_dataset('global_dry_mass_2020_01_aggregated_computed.nc') #updated filename

# %% 
steps = 40
# plot the mass as a function of time
# %%
# plot the mass as a function of prediction_timedelta
# Compute mean and standard deviation
gc_mass_arr = np.array(ds.mass)
mean = np.mean(gc_mass_arr, axis=0)
std_dev = np.std(gc_mass_arr, axis=0)

# %%
# Plot
t = np.linspace(6, 6 * steps, steps)

# %%
plt.plot(t, mean[0:steps], label="Mean", color='blue')
plt.fill_between(t, mean[0:steps] - std_dev[0:steps], mean[0:steps] + std_dev[0:steps], color='blue', alpha=0.2, label="±1 Std Dev")

plt.xlabel('lead time (hr)')
plt.ylabel('mass (kg)')
plt.xticks(np.arange(0, 6 * (steps + 1), 12))
plt.xticks(rotation=90) 
plt.title('Global Dry Air Mass')
plt.legend()
plt.savefig('global_dry_mass_2020_01.png')
# %%
