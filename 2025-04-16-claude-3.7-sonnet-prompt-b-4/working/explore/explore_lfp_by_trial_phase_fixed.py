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

# Define consistent parameters for power spectrum analysis
def compute_power_spectrum(signal_data, fs, nperseg=256):
    if len(signal_data) < nperseg:
        # If segment is too short, pad with zeros
        signal_data = np.pad(signal_data, (0, nperseg - len(signal_data)), 'constant')
    
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=nperseg)
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

# Compute average power spectra with consistent frequency bins
print("Computing power spectra...")

# Use a fixed nperseg value for consistent spectra length
nperseg = 256

# Compute spectra for each trial's data
enc_spectra = []
maint_spectra = []
retr_spectra = []
freq_bins = None  # This will store our frequency bins for plotting

for enc_seg in encoding_data:
    if len(enc_seg) > 0:
        f, Pxx = compute_power_spectrum(enc_seg, sampling_rate, nperseg)
        if freq_bins is None:
            freq_bins = f  # Save the frequency bins for later
        enc_spectra.append(Pxx)

for maint_seg in maintenance_data:
    if len(maint_seg) > 0:
        f, Pxx = compute_power_spectrum(maint_seg, sampling_rate, nperseg)
        maint_spectra.append(Pxx)

for retr_seg in retrieval_data:
    if len(retr_seg) > 0:
        f, Pxx = compute_power_spectrum(retr_seg, sampling_rate, nperseg)
        retr_spectra.append(Pxx)

# Calculate mean spectra if we have data
if enc_spectra and freq_bins is not None:
    mean_enc_spectrum = np.mean(enc_spectra, axis=0)
    mean_maint_spectrum = np.mean(maint_spectra, axis=0) if maint_spectra else np.zeros_like(mean_enc_spectrum)
    mean_retr_spectrum = np.mean(retr_spectra, axis=0) if retr_spectra else np.zeros_like(mean_enc_spectrum)

    # Plot the average spectra for each phase
    plt.figure(figsize=(12, 8))
    # Only plot up to 30 Hz
    mask = freq_bins <= 30
    plt.plot(freq_bins[mask], mean_enc_spectrum[mask], label='Encoding')
    plt.plot(freq_bins[mask], mean_maint_spectrum[mask], label='Maintenance')
    plt.plot(freq_bins[mask], mean_retr_spectrum[mask], label='Retrieval')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectrum by Trial Phase (Channel 4)')
    plt.legend()
    plt.savefig('explore/power_by_phase.png', dpi=300, bbox_inches='tight')

    # Plot relative power (normalized to highlight differences)
    plt.figure(figsize=(12, 8))
    # Normalize by the mean over all phases
    all_mean = (mean_enc_spectrum + mean_maint_spectrum + mean_retr_spectrum) / 3
    # Avoid division by zero
    all_mean[all_mean == 0] = 1
    
    plt.plot(freq_bins[mask], mean_enc_spectrum[mask] / all_mean[mask], label='Encoding')
    plt.plot(freq_bins[mask], mean_maint_spectrum[mask] / all_mean[mask], label='Maintenance')
    plt.plot(freq_bins[mask], mean_retr_spectrum[mask] / all_mean[mask], label='Retrieval')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative Power')
    plt.title('Relative Power Spectrum by Trial Phase (Channel 4)')
    plt.legend()
    plt.savefig('explore/relative_power_by_phase.png', dpi=300, bbox_inches='tight')

# Plot time-frequency representation for a full trial
# Select a representative trial
trial_idx = 5
if trial_idx < len(selected_trials):
    trial = selected_trials.iloc[trial_idx]
    # Extract a segment that spans the entire trial
    trial_start = trial['start_time']
    trial_end = trial['stop_time']
    trial_lfp = extract_lfp_segment(trial_start, trial_end, channel_idx)
    
    if len(trial_lfp) > 0:
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
        event_times = {
            'Fixation': trial['timestamps_FixationCross'] - trial_start,
            'Encoding': trial['timestamps_Encoding1'] - trial_start,
            'Maintenance': trial['timestamps_Maintenance'] - trial_start,
            'Probe': trial['timestamps_Probe'] - trial_start
        }
        
        colors = {'Fixation': 'r', 'Encoding': 'g', 'Maintenance': 'b', 'Probe': 'm'}
        
        for event, time in event_times.items():
            if not np.isnan(time) and time >= 0 and time <= (trial_end - trial_start):
                plt.axvline(time, color=colors[event], linestyle='--', label=event)
        
        plt.colorbar(label='Power (dB)')
        plt.legend()
        plt.savefig('explore/trial_spectrogram.png', dpi=300, bbox_inches='tight')

print("Analysis complete. Plots saved to 'explore' directory")