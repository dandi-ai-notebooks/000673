"""
This script explores the neural data (LFPs and spikes) in the NWB file
from Dandiset 000673, focusing on data visualization and basic analyses.
(Simplified version to handle data structures correctly)
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from matplotlib.gridspec import GridSpec

# Set style for plots
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Get basic information about the LFP data
print("=== LFP DATA INFORMATION ===")
lfp = nwb.acquisition["LFPs"]
print(f"LFP data shape: {lfp.data.shape}")
print(f"LFP sampling rate: {lfp.rate} Hz")

# Get electrode information
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Electrode locations: {electrodes_df['location'].unique()}")

# Get basic information about units (neurons)
print("\n=== UNIT (NEURON) INFORMATION ===")
try:
    units_df = nwb.units.to_dataframe()
    print(f"Number of units: {len(units_df)}")
    
    # Print columns available in units dataframe
    print(f"Columns in units dataframe: {units_df.columns.tolist()}")
    
    # Count spike times
    spike_counts = []
    for _, unit in units_df.iterrows():
        if 'spike_times' in unit and unit['spike_times'] is not None:
            spike_counts.append(len(unit['spike_times']))
        else:
            spike_counts.append(0)
    
    print(f"Total spike count: {sum(spike_counts)}")
    print(f"Mean spikes per unit: {np.mean(spike_counts):.2f}")
    print(f"Max spikes per unit: {np.max(spike_counts)}")
    
except Exception as e:
    print(f"Error processing units: {e}")

# Sample LFP data (10 seconds from a few channels)
print("\nExtracting sample LFP data...")
sample_duration = 10  # seconds
start_time = 100  # start 100 seconds into the recording
start_idx = int(start_time * lfp.rate)
end_idx = start_idx + int(sample_duration * lfp.rate)

# Ensure indices are within range
end_idx = min(end_idx, lfp.data.shape[0])
if end_idx <= start_idx:
    start_idx = 0
    end_idx = min(int(sample_duration * lfp.rate), lfp.data.shape[0])

# Select a few channels for visualization (one from each brain region if possible)
np.random.seed(42)  # for reproducibility

# Map to store channels by region
regions_to_channels = {}
for i, row in electrodes_df.iterrows():
    region = row['location']
    if region not in regions_to_channels:
        regions_to_channels[region] = []
    regions_to_channels[region].append(i)

# Select one channel from each region
selected_channels = []
for region, channels in regions_to_channels.items():
    if channels:
        selected_channels.append(np.random.choice(channels))

# Ensure we don't have too many channels for visualization
if len(selected_channels) > 8:
    selected_channels = np.random.choice(selected_channels, 8, replace=False)

# Extract LFP data for selected channels
print(f"Selected channels: {selected_channels}")
lfp_sample = np.zeros((end_idx - start_idx, len(selected_channels)))

for i, channel_idx in enumerate(selected_channels):
    if channel_idx < lfp.data.shape[1]:
        lfp_sample[:, i] = lfp.data[start_idx:end_idx, channel_idx]
    else:
        print(f"Channel index {channel_idx} out of bounds for LFP data with shape {lfp.data.shape}")

# Create time array
time = np.arange(lfp_sample.shape[0]) / lfp.rate

# Plot LFP data from different brain regions
plt.figure(figsize=(15, 10))
for i, channel_idx in enumerate(selected_channels):
    # Offset each channel for better visualization
    offset = i * np.std(lfp_sample[:, i]) * 3
    if channel_idx < len(electrodes_df):
        location = electrodes_df.iloc[channel_idx]['location']
    else:
        location = "Unknown"
    plt.plot(time, lfp_sample[:, i] + offset, label=f"Ch {channel_idx} ({location})")

plt.xlabel('Time (s)')
plt.ylabel('LFP Amplitude (V) + Offset')
plt.title(f'LFP Traces from Different Brain Regions (t={start_time}s to t={start_time+sample_duration}s)')
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_traces.png', dpi=300)

# Calculate and plot power spectral density for selected channels
plt.figure(figsize=(15, 8))

for i, channel_idx in enumerate(selected_channels):
    # Calculate PSD
    f, psd = signal.welch(lfp_sample[:, i], lfp.rate, nperseg=min(1024, lfp_sample.shape[0]))
    
    # Plot only up to 100 Hz
    mask = f <= 100
    if channel_idx < len(electrodes_df):
        location = electrodes_df.iloc[channel_idx]['location']
    else:
        location = "Unknown"
    plt.semilogy(f[mask], psd[mask], label=f"Ch {channel_idx} ({location})")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.title('Power Spectral Density of LFP Signals')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_psd.png', dpi=300)

# Calculate time-frequency representations (spectrograms) for a few channels
print("\nComputing time-frequency representations...")

# Select a few representative channels (up to 4 for visualization)
if len(selected_channels) > 4:
    representative_channels = selected_channels[:4]
else:
    representative_channels = selected_channels

# Compute spectrograms
fig, axs = plt.subplots(len(representative_channels), 1, figsize=(12, 4*len(representative_channels)), sharex=True)

for i, channel_idx in enumerate(representative_channels):
    f, t, Sxx = signal.spectrogram(lfp_sample[:, selected_channels.index(channel_idx)], 
                                   fs=lfp.rate, nperseg=min(256, lfp_sample.shape[0]//4),
                                   noverlap=min(128, lfp_sample.shape[0]//8))
    
    # Plot spectrogram up to 100 Hz
    f_mask = f <= 100
    if channel_idx < len(electrodes_df):
        location = electrodes_df.iloc[channel_idx]['location']
    else:
        location = "Unknown"
    
    if len(representative_channels) == 1:
        ax = axs
    else:
        ax = axs[i]
    
    pcm = ax.pcolormesh(t, f[f_mask], 10 * np.log10(Sxx[f_mask]), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Channel {channel_idx} ({location})')
    
    # Add colorbar
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label('Power/Frequency (dB/Hz)')

plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_spectrograms.png', dpi=300)

# Close the file
io.close()
f.close()
file.close()

print("\nNeural data exploration completed!")