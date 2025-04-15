# This script explores the Local Field Potential (LFP) data from an NWB file for subject sub-1 session 1.
# It focuses on visualizing the first 10 channels over a specific time window to avoid overloading memory.
# A plot of the LFP signals is saved as an image for further analysis.

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile

# URL to the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"

# Use remfile to handle remote file access
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, "r")
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb_data = io.read()

# Access LFP data
lfp_series = nwb_data.acquisition["LFPs"]

# Extract data for the first 10 channels and first 1000 timepoints
selected_data = lfp_series.data[:1000, :10]

# Create a time array for plotting
time = np.arange(selected_data.shape[0]) / lfp_series.rate

# Plot the selected LFP data
plt.figure(figsize=(15, 6))
for i, channel_data in enumerate(selected_data.T):
    plt.plot(time, channel_data + i*0.1, label=f"Channel {i+1}")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("LFP Data: First 10 Channels")
plt.legend(loc="upper right")
plt.tight_layout()

# Save the plot without displaying it
plot_path = "tmp_scripts/lfp_plot.png"
plt.savefig(plot_path)
plt.close()

# Close the NWB file
io.close()
h5_file.close()