# explore/explore_lfp.py
# This script explores the LFP data in sub-1/sub-1_ses-1_ecephys+image.nwb.
# It plots the distribution of LFP values for a subset of electrodes.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the LFP data
lfps = nwb.acquisition["LFPs"]
data = lfps.data
electrodes = lfps.electrodes.table.to_dataframe()

# Select a subset of electrodes
num_electrodes = min(10, len(electrodes))
electrode_ids = electrodes.index[:num_electrodes]

# Select a subset of time points
num_time_points = min(1000, data.shape[0])

# Extract LFP data for these electrodes
lfp_data = data[:num_time_points, electrode_ids]

# Plot the distribution of LFP values for each electrode using histograms
plt.figure(figsize=(12, 6))
for i, electrode_id in enumerate(electrode_ids):
    plt.subplot(2, 5, i + 1)
    plt.hist(lfp_data[:, i], bins=50)
    plt.title(f"Electrode {electrode_id}")
    plt.xlabel("LFP Value (volts)")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("explore/lfp_distribution.png")
plt.close()