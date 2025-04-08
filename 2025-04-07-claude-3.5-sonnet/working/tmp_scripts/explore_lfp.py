"""
This script explores the LFP data from sub-1_ses-1, focusing on:
1. Example LFP traces
2. Time-frequency analysis around trial events
3. Neural activity during different task phases
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi
from scipy import signal
import seaborn as sns

# Load the data
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get LFP data
lfps = nwb.acquisition["LFPs"]
sampling_rate = lfps.rate  # 400 Hz

# Get example trial timestamps
trials = nwb.intervals["trials"]
example_trial_idx = 0  # Look at first trial
maintenance_start = trials["timestamps_Maintenance"][example_trial_idx]
probe_start = trials["timestamps_Probe"][example_trial_idx]

# Calculate time window around maintenance period
window_before = 1.0  # seconds before maintenance
window_after = 2.0   # seconds after maintenance

# Convert times to indices
start_idx = int((maintenance_start - window_before) * sampling_rate)
end_idx = int((maintenance_start + window_after) * sampling_rate)

# Get LFP data for this window (first few channels)
n_channels = 4
time = np.arange(start_idx, end_idx) / sampling_rate
lfp_data = lfps.data[start_idx:end_idx, :n_channels]

# Create figure
plt.figure(figsize=(15, 10))

# Plot 1: Raw LFP traces
plt.subplot(2, 1, 1)
for i in range(n_channels):
    # Offset each channel for visibility
    offset = i * np.std(lfp_data[:, 0]) * 3
    plt.plot(time - maintenance_start, lfp_data[:, i] + offset, label=f'Channel {i+1}')

plt.axvline(x=0, color='r', linestyle='--', label='Maintenance Start')
plt.axvline(x=probe_start - maintenance_start, color='g', linestyle='--', label='Probe Start')
plt.xlabel('Time relative to maintenance period (s)')
plt.ylabel('LFP amplitude (μV)')
plt.title('Example LFP Traces During Working Memory Maintenance')
plt.legend()

# Plot 2: Time-frequency analysis for first channel
plt.subplot(2, 1, 2)
f, t, Sxx = signal.spectrogram(lfp_data[:, 0], fs=sampling_rate, 
                              nperseg=int(sampling_rate/2),
                              noverlap=int(sampling_rate/4))
plt.pcolormesh(t - window_before, f, 10 * np.log10(Sxx), shading='gouraud')
plt.axvline(x=0, color='r', linestyle='--', label='Maintenance Start')
plt.axvline(x=probe_start - maintenance_start, color='g', linestyle='--', label='Probe Start')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time relative to maintenance period (s)')
plt.title('Time-Frequency Analysis of LFP During Working Memory Maintenance')
plt.colorbar(label='Power (dB)')
plt.ylim(0, 100)  # Focus on frequencies up to 100 Hz

plt.tight_layout()
plt.savefig('tmp_scripts/lfp_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save some statistics about the LFP data
with open('tmp_scripts/lfp_stats.txt', 'w') as f:
    f.write(f"Number of LFP channels: {lfps.data.shape[1]}\n")
    f.write(f"LFP sampling rate: {sampling_rate} Hz\n")
    f.write(f"Total recording duration: {lfps.data.shape[0]/sampling_rate:.2f} seconds\n")
    
    # Calculate mean amplitude by channel
    mean_amplitudes = np.mean(np.abs(lfps.data[:1000, :]), axis=0)  # First 1000 samples
    f.write("\nMean amplitude by channel (first 1000 samples, first 5 channels):\n")
    for i, amp in enumerate(mean_amplitudes[:5]):
        f.write(f"Channel {i+1}: {amp:.2f} μV\n")