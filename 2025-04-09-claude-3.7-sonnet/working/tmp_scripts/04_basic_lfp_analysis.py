"""
This script focuses on exploring the LFP data in the NWB file from Dandiset 000673,
creating visualizations of the signals, power spectra, and time-frequency representations.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# Set style for plots
sns.set_theme()

# Load the NWB file
print("Loading NWB file...")
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
print(f"Number of electrodes: {len(electrodes_df)}")

# Count electrodes by brain region
regions = electrodes_df['location'].value_counts()
print("\nElectrodes by brain region:")
for region, count in regions.items():
    print(f"  {region}: {count}")

# Sample LFP data (5 seconds from various brain regions)
print("\nExtracting sample LFP data...")
sample_duration = 5  # seconds
start_time = 200  # seconds into the recording
start_idx = int(start_time * lfp.rate)
end_idx = start_idx + int(sample_duration * lfp.rate)

# Ensure indices are within range
end_idx = min(end_idx, lfp.data.shape[0])
if end_idx <= start_idx:
    start_idx = 0
    end_idx = min(int(sample_duration * lfp.rate), lfp.data.shape[0])

# Select representative channels from different brain regions
region_names = electrodes_df['location'].unique()
selected_channels = []
channel_regions = []  # Keep track of the region for each selected channel

# For each region, try to get one representative electrode
for region in region_names:
    region_electrodes = electrodes_df[electrodes_df['location'] == region].index.tolist()
    if region_electrodes:
        # Choose the middle electrode from this region
        middle_idx = len(region_electrodes) // 2
        selected_channels.append(region_electrodes[middle_idx])
        channel_regions.append(region)

# Limit to at most 6 channels for better visualization
if len(selected_channels) > 6:
    indices = np.linspace(0, len(selected_channels)-1, 6, dtype=int)
    selected_channels = [selected_channels[i] for i in indices]
    channel_regions = [channel_regions[i] for i in indices]

print(f"Selected channels: {selected_channels}")
print(f"Corresponding regions: {channel_regions}")

# Extract LFP data for selected channels
lfp_sample = lfp.data[start_idx:end_idx, selected_channels]

# Create time array
time = np.arange(lfp_sample.shape[0]) / lfp.rate

# Plot LFP traces
plt.figure(figsize=(15, 10))

for i, (channel_idx, region) in enumerate(zip(selected_channels, channel_regions)):
    # Normalize and offset each signal for better visualization
    signal_data = lfp_sample[:, i]
    normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    offset = i * 6  # Increase vertical separation
    plt.plot(time, normalized + offset, linewidth=1, label=f"Ch {channel_idx} ({region})")

plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude + Offset')
plt.title(f'LFP Traces from Different Brain Regions (t={start_time} to t={start_time+sample_duration}s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yticks([])  # Hide y-axis ticks as they're not meaningful with offsets
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_traces_normalized.png', dpi=300)

# Calculate and plot power spectral density for each channel
plt.figure(figsize=(15, 8))

for i, (channel_idx, region) in enumerate(zip(selected_channels, channel_regions)):
    # Calculate PSD
    f, psd = signal.welch(lfp_sample[:, i], lfp.rate, nperseg=min(1024, lfp_sample.shape[0]))
    
    # Plot only up to 100 Hz
    mask = f <= 100
    plt.semilogy(f[mask], psd[mask], label=f"Ch {channel_idx} ({region})")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.title('Power Spectral Density of LFP Signals from Different Brain Regions')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_psd_by_region.png', dpi=300)

# Create spectrograms for selected channels
print("\nComputing spectrograms...")

# Determine number of rows and columns for subplots
n_channels = len(selected_channels)
n_rows = 3
n_cols = (n_channels + n_rows - 1) // n_rows  # Ceiling division

fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True)
axs = axs.flatten()

for i, (channel_idx, region) in enumerate(zip(selected_channels, channel_regions)):
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(
        lfp_sample[:, i], 
        fs=lfp.rate, 
        nperseg=min(256, lfp_sample.shape[0]//4),
        noverlap=min(128, lfp_sample.shape[0]//8)
    )
    
    # Plot spectrogram up to 100 Hz
    f_mask = f <= 100
    pcm = axs[i].pcolormesh(t, f[f_mask], 10 * np.log10(Sxx[f_mask]), 
                          shading='gouraud', cmap='viridis')
    axs[i].set_ylabel('Frequency (Hz)')
    axs[i].set_title(f'Channel {channel_idx} ({region})')
    
    # Add colorbar
    cb = fig.colorbar(pcm, ax=axs[i])
    cb.set_label('Power/Frequency (dB/Hz)')

# Set common x-axis label
for ax in axs[-n_cols:]:
    ax.set_xlabel('Time (s)')

# Hide unused subplots if any
for j in range(i+1, len(axs)):
    axs[j].set_visible(False)

plt.tight_layout()
plt.savefig('tmp_scripts/lfp_spectrograms_by_region.png', dpi=300)

# Look at a specific frequency band (theta: 4-8 Hz)
print("\nExtracting theta band power...")

# Filter data to extract theta band (4-8 Hz)
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# Filter each channel
theta_data = np.zeros_like(lfp_sample)
for i in range(lfp_sample.shape[1]):
    theta_data[:, i] = bandpass_filter(lfp_sample[:, i], 4, 8, lfp.rate)

# Plot filtered signals
plt.figure(figsize=(15, 10))

for i, (channel_idx, region) in enumerate(zip(selected_channels, channel_regions)):
    # Normalize and offset each signal for better visualization
    signal_data = theta_data[:, i]
    normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    offset = i * 6  # Increase vertical separation
    plt.plot(time, normalized + offset, linewidth=1, label=f"Ch {channel_idx} ({region})")

plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude + Offset')
plt.title('Theta Band (4-8 Hz) LFP Activity from Different Brain Regions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yticks([])  # Hide y-axis ticks as they're not meaningful with offsets
plt.tight_layout()
plt.savefig('tmp_scripts/theta_band_by_region.png', dpi=300)

# Calculate theta power over time using the Hilbert transform
print("\nCalculating theta power envelope...")

theta_power = np.zeros_like(theta_data)
for i in range(theta_data.shape[1]):
    analytic_signal = signal.hilbert(theta_data[:, i])
    amplitude_envelope = np.abs(analytic_signal)
    theta_power[:, i] = amplitude_envelope

# Plot theta power
plt.figure(figsize=(15, 10))

for i, (channel_idx, region) in enumerate(zip(selected_channels, channel_regions)):
    # Normalize and offset each power trace for better visualization
    power_data = theta_power[:, i]
    normalized = (power_data - np.mean(power_data)) / np.std(power_data)
    offset = i * 6  # Increase vertical separation
    plt.plot(time, normalized + offset, linewidth=1, label=f"Ch {channel_idx} ({region})")

plt.xlabel('Time (s)')
plt.ylabel('Normalized Power + Offset')
plt.title('Theta Band Power Over Time from Different Brain Regions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yticks([])  # Hide y-axis ticks as they're not meaningful with offsets
plt.tight_layout()
plt.savefig('tmp_scripts/theta_power_by_region.png', dpi=300)

# Close files
io.close()
f.close()
file.close()

print("\nLFP analysis completed!")