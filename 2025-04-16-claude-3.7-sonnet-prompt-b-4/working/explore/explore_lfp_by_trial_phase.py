"""
This script explores how LFP activity changes across different phases of the working memory task.
We'll extract LFP data during encoding, maintenance, and retrieval phases, and analyze
spectral differences between these phases.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import remfile
import h5py
import pynwb
import pandas as pd

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
sampling_rate = lfp.rate  # 400 Hz

# Get trials data
trials = nwb.intervals['trials']
trials_df = trials.to_dataframe()

# Pick a representative channel for analysis (using channel 4 which showed high amplitude in our first analysis)
channel_idx = 3  # Index for the 4th channel

# Function to extract LFP data for a specific time window
def extract_lfp_segment(start_time, end_time, channel_idx):
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)
    if start_idx < 0:
        start_idx = 0
    if end_idx > lfp_data.shape[0]:
        end_idx = lfp_data.shape[0]
    return lfp_data[start_idx:end_idx, channel_idx]

# Function to calculate power spectrum
def compute_power_spectrum(signal_data, fs):
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(1024, len(signal_data)))
    return f, Pxx

# Select the first 10 trials for analysis
num_trials = 10
selected_trials = trials_df.iloc[:num_trials]

# Prepare data structures for each phase
encoding_data = []
maintenance_data = []
retrieval_data = []

# Extract LFP segments for each phase
for idx, trial in selected_trials.iterrows():
    # Encoding phase (first image)
    enc_start = trial['timestamps_Encoding1']
    enc_end = trial['timestamps_Encoding1_end']
    if not np.isnan(enc_start) and not np.isnan(enc_end):
        encoding_data.append(extract_lfp_segment(enc_start, enc_end, channel_idx))
    
    # Maintenance phase
    maint_start = trial['timestamps_Maintenance']
    # Use a fixed window of 1 second during maintenance
    maint_end = maint_start + 1.0
    if not np.isnan(maint_start):
        maintenance_data.append(extract_lfp_segment(maint_start, maint_end, channel_idx))
    
    # Retrieval phase (probe)
    retr_start = trial['timestamps_Probe']
    # Use a fixed window of 1 second for retrieval
    retr_end = retr_start + 1.0
    if not np.isnan(retr_start):
        retrieval_data.append(extract_lfp_segment(retr_start, retr_end, channel_idx))

# Compute average power spectra
enc_spectra = []
maint_spectra = []
retr_spectra = []

print("Computing power spectra...")

# Compute spectra for each trial's data
for enc_seg in encoding_data:
    if len(enc_seg) > 0:
        f, Pxx = compute_power_spectrum(enc_seg, sampling_rate)
        enc_spectra.append(Pxx)

for maint_seg in maintenance_data:
    if len(maint_seg) > 0:
        f, Pxx = compute_power_spectrum(maint_seg, sampling_rate)
        maint_spectra.append(Pxx)

for retr_seg in retrieval_data:
    if len(retr_seg) > 0:
        f, Pxx = compute_power_spectrum(retr_seg, sampling_rate)
        retr_spectra.append(Pxx)

# Calculate mean spectra
mean_enc_spectrum = np.mean(np.array(enc_spectra), axis=0) if enc_spectra else np.zeros_like(f)
mean_maint_spectrum = np.mean(np.array(maint_spectra), axis=0) if maint_spectra else np.zeros_like(f)
mean_retr_spectrum = np.mean(np.array(retr_spectra), axis=0) if retr_spectra else np.zeros_like(f)

# Plot the average spectra for each phase
plt.figure(figsize=(12, 8))
# Only plot up to 30 Hz
mask = f <= 30
plt.plot(f[mask], mean_enc_spectrum[mask], label='Encoding')
plt.plot(f[mask], mean_maint_spectrum[mask], label='Maintenance')
plt.plot(f[mask], mean_retr_spectrum[mask], label='Retrieval')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectrum by Trial Phase (Channel 4)')
plt.legend()
plt.savefig('explore/power_by_phase.png', dpi=300, bbox_inches='tight')

# Plot time-frequency representation for a full trial
# Select a representative trial
trial_idx = 5
if trial_idx < len(selected_trials):
    trial = selected_trials.iloc[trial_idx]
    # Extract a segment that spans the entire trial
    trial_start = trial['start_time']
    trial_end = trial['stop_time']
    trial_lfp = extract_lfp_segment(trial_start, trial_end, channel_idx)
    
    # Compute time-frequency representation using spectrogram
    f, t, Sxx = signal.spectrogram(trial_lfp, fs=sampling_rate, nperseg=128, noverlap=64)
    
    # Plot spectrogram
    plt.figure(figsize=(12, 8))
    mask = f <= 30  # Only show up to 30 Hz
    plt.pcolormesh(t, f[mask], 10 * np.log10(Sxx[mask, :]), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.title(f'LFP Spectrogram for Trial {trial_idx+1} (Channel 4)')
    
    # Add markers for trial events
    plt.axvline(trial['timestamps_FixationCross'] - trial_start, color='r', linestyle='--', label='Fixation')
    plt.axvline(trial['timestamps_Encoding1'] - trial_start, color='g', linestyle='--', label='Encoding Start')
    plt.axvline(trial['timestamps_Maintenance'] - trial_start, color='b', linestyle='--', label='Maintenance')
    plt.axvline(trial['timestamps_Probe'] - trial_start, color='m', linestyle='--', label='Probe')
    plt.colorbar(label='Power (dB)')
    plt.legend()
    plt.savefig('explore/trial_spectrogram.png', dpi=300, bbox_inches='tight')

print("Analysis complete. Plots saved to 'explore' directory")