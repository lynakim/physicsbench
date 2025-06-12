# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %%
# Open and concatenate datasets for all 12 months
datasets = []
for i in range(1, 10):
    ds = xr.open_dataset('global_dry_mass_2020_0' + str(i) + '.nc')
    datasets.append(ds)

for i in range(10, 13):
    ds = xr.open_dataset('global_dry_mass_2020_' + str(i) + '.nc')
    datasets.append(ds)

# Concatenate all datasets into a single array
mass = np.concatenate([ds['mass'].values for ds in datasets], axis=0)

# Get the lead times from the first dataset (they should be the same for all)
t = datasets[0]['prediction_timedelta'].values.astype('timedelta64[h]').astype(float)

# Compute the mean across all time entries (axis=0)
mean_mass = np.mean(mass, axis=0)  # This gives us shape (40,)
# Compute the standard deviation across all time entries (axis=0)
std_mass = np.std(mass, axis=0)    # This gives us shape (40,)

# Print shapes for debugging
print("Shape of t:", t.shape)
print("Shape of mean_mass:", mean_mass.shape)

# %%
plt.figure(figsize=(12, 6))
plt.plot(t, mean_mass, label="Mean", color='blue')
plt.fill_between(t, mean_mass - std_mass, mean_mass + std_mass, color='blue', alpha=0.2, label="±1 Std Dev")

plt.xlabel('lead time (hr)')
plt.ylabel('mass (kg)')
plt.xticks(np.arange(0, 241, 12))
plt.xticks(rotation=90)
plt.title('GC, Global Dry Air Mass 2020 (All Months)')
plt.legend()
plt.grid(True)
plt.show()

# %%
