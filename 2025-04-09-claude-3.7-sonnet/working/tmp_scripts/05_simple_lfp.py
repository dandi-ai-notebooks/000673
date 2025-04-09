"""
This script provides a simplified analysis of LFP data from Dandiset 000673,
focusing on visualizing signals from the first few channels.
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

# Sample LFP data (5 seconds from the first few channels)
print("\nExtracting sample LFP data...")
sample_duration = 5  # seconds
start_time = 200  # seconds into the recording
start_idx = int(start_time * lfp.rate)
end_idx = start_idx + int(sample_duration * lfp.rate)

# Ensure indices are within range
data_shape = lfp.data.shape
end_idx = min(end_idx, data_shape[0])
if end_idx <= start_idx:
    start_idx = 0
    end_idx = min(int(sample_duration * lfp.rate), data_shape[0])

# Use only the first 6 channels to avoid indexing issues
num_channels = min(6, data_shape[1])
selected_channels = list(range(num_channels))

print(f"Using first {num_channels} channels")
print(f"Time window: {start_idx/lfp.rate:.2f}s to {end_idx/lfp.rate:.2f}s")

# Extract LFP data
lfp_sample = np.zeros((end_idx - start_idx, num_channels))
for i in range(num_channels):
    lfp_sample[:, i] = lfp.data[start_idx:end_idx, i]

# Create time array
time = np.arange(lfp_sample.shape[0]) / lfp.rate

# Get brain regions for these channels if possible
channel_regions = []
for i in range(num_channels):
    if i < len(electrodes_df):
        channel_regions.append(electrodes_df.iloc[i]['location'])
    else:
        channel_regions.append(f"Unknown")

# Plot LFP traces
plt.figure(figsize=(15, 10))

for i in range(num_channels):
    # Normalize and offset each signal for better visualization
    signal_data = lfp_sample[:, i]
    normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    offset = i * 6  # Increase vertical separation
    plt.plot(time, normalized + offset, linewidth=1, label=f"Ch {i} ({channel_regions[i]})")

plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude + Offset')
plt.title(f'LFP Traces (t={start_time} to t={start_time+sample_duration}s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yticks([])  # Hide y-axis ticks as they're not meaningful with offsets
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_traces_simple.png', dpi=300)

# Calculate and plot power spectral density for each channel
plt.figure(figsize=(15, 8))

for i in range(num_channels):
    # Calculate PSD
    f, psd = signal.welch(lfp_sample[:, i], lfp.rate, nperseg=min(1024, lfp_sample.shape[0]))
    
    # Plot only up to 100 Hz
    mask = f <= 100
    plt.semilogy(f[mask], psd[mask], label=f"Ch {i} ({channel_regions[i]})")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.title('Power Spectral Density of LFP Signals')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_psd_simple.png', dpi=300)

# Create spectrograms for selected channels
print("\nComputing spectrograms...")

fig, axs = plt.subplots(num_channels, 1, figsize=(12, 4*num_channels), sharex=True)

for i in range(num_channels):
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(
        lfp_sample[:, i], 
        fs=lfp.rate, 
        nperseg=min(256, lfp_sample.shape[0]//4),
        noverlap=min(128, lfp_sample.shape[0]//8)
    )
    
    # Plot spectrogram up to 100 Hz
    f_mask = f <= 100
    ax = axs[i] if num_channels > 1 else axs
    pcm = ax.pcolormesh(t, f[f_mask], 10 * np.log10(Sxx[f_mask]), 
                       shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Channel {i} ({channel_regions[i]})')
    
    # Add colorbar
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label('Power/Frequency (dB/Hz)')

# Set common x-axis label
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_spectrograms_simple.png', dpi=300)

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
for i in range(num_channels):
    theta_data[:, i] = bandpass_filter(lfp_sample[:, i], 4, 8, lfp.rate)

# Plot filtered signals
plt.figure(figsize=(15, 10))

for i in range(num_channels):
    # Normalize and offset each signal for better visualization
    signal_data = theta_data[:, i]
    normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    offset = i * 6  # Increase vertical separation
    plt.plot(time, normalized + offset, linewidth=1, label=f"Ch {i} ({channel_regions[i]})")

plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude + Offset')
plt.title('Theta Band (4-8 Hz) LFP Activity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yticks([])  # Hide y-axis ticks as they're not meaningful with offsets
plt.tight_layout()
plt.savefig('tmp_scripts/theta_band_simple.png', dpi=300)

# Close files
io.close()
f.close()
file.close()

print("\nLFP analysis completed!")