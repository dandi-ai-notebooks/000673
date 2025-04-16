"""
This script explores the LFP (Local Field Potential) data from the NWB file.
We'll examine the structure of the LFP data, plot some example traces, 
and analyze the frequency content.
"""

import numpy as np
import matplotlib.pyplot as plt
import remfile
import h5py
import pynwb

# Set the file URL
url = "https://api.dandiarchive.org/api/assets/8b91e132-3477-43f8-8ec7-4e45fda87fea/download/"

# Open the file
print("Loading NWB file...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data
print("Extracting LFP data...")
lfp = nwb.acquisition["LFPs"]
lfp_data = lfp.data

# Get only a subset of the data to make the plotting more manageable
# Let's get 5 seconds of data (2000 samples at 400 Hz) for all channels
start_time = 100  # Start at 100 seconds into the recording
duration = 5  # 5 seconds of data
start_index = int(start_time * lfp.rate)
end_index = int((start_time + duration) * lfp.rate)

# Extract the subset
lfp_subset = lfp_data[start_index:end_index, :]

# Get electrode information
electrodes_df = lfp.electrodes.table.to_dataframe()
print(f"LFP data shape: {lfp_data.shape}")
print(f"Electrode information:\n{electrodes_df}")
print(f"Sampling rate: {lfp.rate} Hz")

# Plot LFP traces
plt.figure(figsize=(12, 8))
time = np.arange(len(lfp_subset)) / lfp.rate
for i in range(lfp_subset.shape[1]):
    plt.plot(time, lfp_subset[:, i] + i*0.0005, label=f"Channel {electrodes_df['origChannel'].values[i]}")

plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V) + offset')
plt.title('LFP Traces (5-second sample)')
plt.legend()
plt.savefig('explore/lfp_traces.png', dpi=300, bbox_inches='tight')

# Plot power spectrum
plt.figure(figsize=(10, 6))
for i in range(lfp_subset.shape[1]):
    f, Pxx = np.fft.rfftfreq(len(lfp_subset), d=1/lfp.rate), np.abs(np.fft.rfft(lfp_subset[:, i]))**2
    # Only plot up to 100 Hz
    mask = f <= 100
    plt.plot(f[mask], Pxx[mask], label=f"Channel {electrodes_df['origChannel'].values[i]}")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum of LFP Signals')
plt.legend()
plt.savefig('explore/lfp_power_spectrum.png', dpi=300, bbox_inches='tight')
print("Plots saved to 'explore' directory")