"""
This script explores the neural data (LFPs and spikes) in the NWB file
from Dandiset 000673, focusing on data visualization and basic analyses.
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
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")

# Function to get electrode location for a unit
def get_electrode_location(unit_row):
    if 'electrodes' in unit_row and not pd.isna(unit_row['electrodes']):
        try:
            electrode_idx = unit_row['electrodes']
            # Handle if it's an object with proper indexing
            if hasattr(electrode_idx, 'item'):
                electrode_idx = electrode_idx.item()
            return electrodes_df.iloc[electrode_idx]['location']
        except:
            return 'Unknown'
    return 'Unknown'

# Add location information to units
units_df['location'] = units_df.apply(get_electrode_location, axis=1)

# Count units by brain region
unit_counts = units_df['location'].value_counts()
print("\nUnits by brain region:")
for region, count in unit_counts.items():
    print(f"  {region}: {count}")

# Sample LFP data (10 seconds from random channels)
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

# Randomly select channels from different brain regions
np.random.seed(42)  # for reproducibility
regions = electrodes_df['location'].unique()
selected_channels = []

for region in regions:
    region_channels = electrodes_df[electrodes_df['location'] == region].index.tolist()
    if region_channels:
        selected_channels.append(np.random.choice(region_channels))

# Ensure we don't have too many channels for visualization
if len(selected_channels) > 8:
    selected_channels = np.random.choice(selected_channels, 8, replace=False)

# Extract LFP data for selected channels
lfp_sample = lfp.data[start_idx:end_idx, selected_channels]

# Create time array
time = np.arange(lfp_sample.shape[0]) / lfp.rate

# Plot LFP data
plt.figure(figsize=(15, 10))
for i, channel_idx in enumerate(selected_channels):
    # Offset each channel for better visualization
    offset = i * np.std(lfp_sample[:, i]) * 3
    location = electrodes_df.iloc[channel_idx]['location']
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
    location = electrodes_df.iloc[channel_idx]['location']
    plt.semilogy(f[mask], psd[mask], label=f"Ch {channel_idx} ({location})")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.title('Power Spectral Density of LFP Signals')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_psd.png', dpi=300)

# Extract time-frequency representations for a subset of channels
print("\nComputing time-frequency representations...")

# Select one representative channel from each brain region (limit to 4 for visualization)
representative_channels = []
for region in regions:
    region_channels = electrodes_df[electrodes_df['location'] == region].index.tolist()
    if region_channels:
        representative_channels.append(np.random.choice(region_channels))
        if len(representative_channels) >= 4:
            break

# Compute spectrograms
fig, axs = plt.subplots(len(representative_channels), 1, figsize=(12, 4*len(representative_channels)), sharex=True)

for i, channel_idx in enumerate(representative_channels):
    f, t, Sxx = signal.spectrogram(lfp_sample[:, selected_channels.index(channel_idx)], 
                                   fs=lfp.rate, nperseg=min(256, lfp_sample.shape[0]//4),
                                   noverlap=min(128, lfp_sample.shape[0]//8))
    
    # Plot spectrogram up to 100 Hz
    f_mask = f <= 100
    location = electrodes_df.iloc[channel_idx]['location']
    
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

# Explore spike data
print("\nExploring spike data...")

# Get information about number of spikes per unit
spike_counts = []
for idx, unit in units_df.iterrows():
    if 'spike_times' in unit and unit['spike_times'] is not None:
        spike_counts.append(len(unit['spike_times']))
    else:
        spike_counts.append(0)

units_df['spike_count'] = spike_counts

print(f"Total spike count: {sum(spike_counts)}")
print(f"Mean spikes per unit: {np.mean(spike_counts):.2f}")
print(f"Max spikes per unit: {np.max(spike_counts)}")

# Plot spike count distribution by brain region
plt.figure(figsize=(12, 8))
sns.boxplot(x='location', y='spike_count', data=units_df)
plt.xticks(rotation=45, ha='right')
plt.title('Spike Count Distribution by Brain Region')
plt.tight_layout()
plt.savefig('tmp_scripts/spike_counts_by_region.png', dpi=300)

# Create raster plot and PSTH for units with highest spike counts
top_units = units_df.nlargest(5, 'spike_count')

# Find trial onsets
if hasattr(nwb, 'trials') and len(nwb.trials) > 0:
    # Get trial start times and memory loads
    trial_starts = nwb.trials['timestamps_Encoding1'][:]
    memory_loads = nwb.trials['loads'][:]
    
    # Create figure for raster plot and PSTH
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(len(top_units), 2, width_ratios=[3, 1], figure=fig)
    
    # Plot for each of the top units
    for i, (idx, unit) in enumerate(top_units.iterrows()):
        spike_times = unit['spike_times']
        unit_location = unit['location']
        
        # Create raster plot
        ax_raster = fig.add_subplot(gs[i, 0])
        
        # Find spikes aligned to trial starts
        # Let's look at spikes from -1s to +6s around trial onset
        pre_time = 1  # seconds before trial start
        post_time = 6  # seconds after trial start
        
        # For each trial
        for j, trial_start in enumerate(trial_starts[:20]):  # limit to first 20 trials for clarity
            # Find spikes around this trial start
            trial_spikes = spike_times[(spike_times >= trial_start - pre_time) & 
                                      (spike_times <= trial_start + post_time)]
            
            # Plot spikes relative to trial start
            if len(trial_spikes) > 0:
                rel_spike_times = trial_spikes - trial_start
                ax_raster.vlines(rel_spike_times, j - 0.4, j + 0.4)
        
        # Add vertical line at time zero (trial onset)
        ax_raster.axvline(x=0, color='r', linestyle='--')
        
        # Add labels
        ax_raster.set_xlabel('Time from Trial Onset (s)')
        ax_raster.set_ylabel('Trial Number')
        ax_raster.set_title(f'Unit {idx} Spikes, Region: {unit_location}')
        ax_raster.set_xlim(-pre_time, post_time)
        
        # Create PSTH (peri-stimulus time histogram)
        ax_psth = fig.add_subplot(gs[i, 1])
        
        # Collect spikes across all trials
        all_rel_spikes = []
        for trial_start in trial_starts:
            trial_spikes = spike_times[(spike_times >= trial_start - pre_time) & 
                                      (spike_times <= trial_start + post_time)]
            if len(trial_spikes) > 0:
                all_rel_spikes.extend(trial_spikes - trial_start)
        
        # Create histogram
        if all_rel_spikes:
            ax_psth.hist(all_rel_spikes, bins=30, range=(-pre_time, post_time), 
                       orientation='horizontal', histtype='step', linewidth=2)
            ax_psth.axhline(y=0, color='r', linestyle='--')
            ax_psth.set_xlabel('Spike Count')
            ax_psth.set_title('PSTH')
            ax_psth.set_ylim(-pre_time, post_time)
        else:
            ax_psth.text(0.5, 0.5, 'No spikes', ha='center', va='center', transform=ax_psth.transAxes)
    
    plt.tight_layout()
    plt.savefig('tmp_scripts/spike_raster_psth.png', dpi=300)

# Get spike waveforms if available
print("\nExploring spike waveforms...")
waveforms_available = []
for idx, unit in units_df.iterrows():
    if 'waveforms' in unit and unit['waveforms'] is not None and len(unit['waveforms']) > 0:
        waveforms_available.append(True)
    else:
        waveforms_available.append(False)

units_df['has_waveforms'] = waveforms_available
print(f"Units with waveforms: {sum(waveforms_available)} out of {len(units_df)}")

# Plot example waveforms from different brain regions
if sum(waveforms_available) > 0:
    # Group units by region
    region_units = units_df[units_df['has_waveforms']].groupby('location')
    
    # Create figure with subplots for each region (up to 6 regions)
    regions_to_plot = min(6, len(region_units))
    fig, axs = plt.subplots(regions_to_plot, 1, figsize=(12, 3*regions_to_plot), sharex=True)
    
    # If only one region, convert axs to a list
    if regions_to_plot == 1:
        axs = [axs]
    
    # For each region, plot waveforms from a sample unit
    for i, (region, group) in enumerate(region_units):
        if i >= regions_to_plot:
            break
            
        # Get a unit with waveforms
        sample_unit = group.iloc[0]
        waveforms = sample_unit['waveforms']
        
        # Plot mean waveform and individual waveforms
        time_points = np.arange(waveforms.shape[1])
        
        # Plot individual waveforms (up to 50 for clarity)
        max_waveforms = min(50, waveforms.shape[0])
        for j in range(max_waveforms):
            axs[i].plot(time_points, waveforms[j], color='gray', alpha=0.3, linewidth=0.5)
        
        # Plot mean waveform
        mean_waveform = np.mean(waveforms, axis=0)
        axs[i].plot(time_points, mean_waveform, color='black', linewidth=2)
        
        axs[i].set_title(f'Unit Waveforms from {region}')
        axs[i].set_ylabel('Amplitude')
    
    axs[-1].set_xlabel('Time (samples)')
    plt.tight_layout()
    plt.savefig('tmp_scripts/spike_waveforms.png', dpi=300)

# Close the file
io.close()
f.close()
file.close()

print("\nNeural data exploration completed!")