import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Plot LFP data
lfp_data = nwb.acquisition["LFPs"].data
lfp_rate = nwb.acquisition["LFPs"].rate
num_channels = lfp_data.shape[1]
time = np.arange(0, 10, 1 / lfp_rate)  # Time in seconds for first 10 seconds
num_timepoints = len(time)

for i in range(min(5, num_channels)): # Plot the first 5 channels
  plt.figure(figsize=(10, 2))
  plt.plot(time, lfp_data[0:num_timepoints, i])
  plt.xlabel("Time (s)")
  plt.ylabel("LFP (volts)")
  plt.title(f"LFP Data - Channel {i}")
  plt.savefig(f"tmp_scripts/lfp_channel_{i}.png")  # Saves the plot to a PNG file
  plt.close()