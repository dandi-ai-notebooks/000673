"""
This script explores the LFP data in Dandiset 000673 (sub-10).
The goal is to understand the properties of the LFP recordings and visualize them.
"""

import numpy as np
import pandas as pd
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.gridspec import GridSpec

# Load
url = "https://api.dandiarchive.org/api/assets/c03df798-01fc-4023-ab51-e1721e2db93c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode information
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Total number of electrodes: {len(electrodes_df)}")
print(f"\nElectrode columns: {list(electrodes_df.columns)}")
print("\nBrain regions:")
print(electrodes_df['location'].value_counts())

# Get a sample of LFP data - limit to first 20 seconds and 5 channels to keep the data size manageable
lfp = nwb.acquisition["LFPs"]
sampling_rate = lfp.rate
print(f"\nLFP sampling rate: {sampling_rate} Hz")

# Get the first 20 seconds (20 * sampling_rate samples) for the first 5 channels
time_window = 20  # seconds
time_samples = int(time_window * sampling_rate)
channel_count = 5

# Get the subset of LFP data
lfp_data = lfp.data[:time_samples, :channel_count]
lfp_time = np.arange(lfp_data.shape[0]) / sampling_rate + lfp.starting_time

# Get the channel locations for these electrodes
channel_indices = lfp.electrodes.data[:channel_count]
channel_locations = electrodes_df.iloc[channel_indices]['location'].values

print(f"\nData shape: {lfp_data.shape}")
print(f"Time range: {lfp_time[0]:.2f} to {lfp_time[-1]:.2f} seconds")
print(f"Selected channel locations: {channel_locations}")

# Plot the LFP traces
plt.figure(figsize=(14, 10))
for i in range(channel_count):
    plt.subplot(channel_count, 1, i+1)
    plt.plot(lfp_time, lfp_data[:, i], linewidth=1)
    plt.title(f'LFP Channel {i} - {channel_locations[i]}')
    if i == channel_count - 1:
        plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
plt.tight_layout()
plt.savefig('explore/lfp_traces.png')

# Compute power spectral density for one channel (first channel)
channel_idx = 0
f, Pxx = signal.welch(lfp_data[:, channel_idx], fs=sampling_rate, nperseg=1024)
# Limit to frequencies below 100 Hz where most relevant LFP activity occurs
freq_mask = f <= 100
f = f[freq_mask]
Pxx = Pxx[freq_mask]

# Plot PSD
plt.figure(figsize=(10, 6))
plt.semilogy(f, Pxx)
plt.title(f'Power Spectral Density - Channel {channel_idx} ({channel_locations[channel_idx]})')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V^2/Hz)')
plt.grid(True)
plt.savefig('explore/lfp_psd.png')

# Now let's compute a time-frequency representation (spectrogram) for one channel
channel_idx = 0
f, t, Sxx = signal.spectrogram(lfp_data[:, channel_idx], fs=sampling_rate, nperseg=256, noverlap=128)
# Limit to frequencies below 100 Hz
freq_mask = f <= 100
f = f[freq_mask]
Sxx = Sxx[freq_mask]

# Plot spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.title(f'Spectrogram - Channel {channel_idx} ({channel_locations[channel_idx]})')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.savefig('explore/lfp_spectrogram.png')

# Finally, let's compute band power in different frequency bands
bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}

# Create a function to calculate band power
def bandpower(data, sf, band):
    f, Pxx = signal.welch(data, fs=sf, nperseg=1024)
    ind_min = np.argmax(f > band[0]) - 1
    ind_max = np.argmax(f > band[1]) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

# Calculate band power for each channel
band_power = np.zeros((channel_count, len(bands)))
for i, (name, band) in enumerate(bands.items()):
    for ch in range(channel_count):
        band_power[ch, i] = bandpower(lfp_data[:, ch], sampling_rate, band)

# Plot band power for each channel
plt.figure(figsize=(12, 8))
bar_width = 0.15
index = np.arange(len(bands))
for ch in range(channel_count):
    plt.bar(index + ch * bar_width, band_power[ch], bar_width, 
            label=f'Channel {ch} ({channel_locations[ch]})')
    
plt.xlabel('Frequency Band')
plt.ylabel('Band Power')
plt.title('LFP Band Power by Channel')
plt.xticks(index + bar_width * (channel_count - 1) / 2, bands.keys())
plt.legend()
plt.tight_layout()
plt.savefig('explore/lfp_band_power.png')

# Close file
h5_file.close()