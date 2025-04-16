"""
Script to plot Local Field Potential (LFP) data from the NWB file using a subset of electrodes.
"""

import matplotlib.pyplot as plt
import numpy as np
import remfile
import h5py
import pynwb

# Load the remote NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access the LFP data
lfp_data = nwb.acquisition['LFPs'].data[:10, :10]  # Access a subset of data for visualization

# Plotting
plt.figure(figsize=(10, 6))
for i in range(lfp_data.shape[1]):
    plt.plot(lfp_data[:, i], label=f'Channel {i}')

plt.title('LFP Signals from Select Electrodes')
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.legend(loc='upper right')

# Save the plot
plt.savefig('explore/lfp_plot.png')
plt.close()