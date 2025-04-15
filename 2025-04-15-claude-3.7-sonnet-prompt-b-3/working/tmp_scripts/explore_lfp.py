# This script explores and visualizes LFP data from different brain regions

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set seaborn style
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode information
electrode_df = nwb.electrodes.to_dataframe()

# Get LFP data
lfps = nwb.acquisition["LFPs"]
print(f"LFP data shape: {lfps.data.shape}")
print(f"Number of time points: {lfps.data.shape[0]}")
print(f"Number of channels: {lfps.data.shape[1]}")
print(f"Sampling rate: {lfps.rate} Hz")

# Get electrode indices and match with electrodes table
electrode_indices = lfps.electrodes.data[:]
electrode_info = electrode_df.iloc[electrode_indices]

# Count electrodes by brain region
region_counts = electrode_info['location'].value_counts()
print("\nElectrode distribution by brain region:")
for region, count in region_counts.items():
    print(f"  {region}: {count} electrodes")

# Select data from different brain regions for visualization
regions_to_plot = [
    'hippocampus_left',
    'hippocampus_right',
    'amygdala_left', 
    'amygdala_right',
    'dorsal_anterior_cingulate_cortex_left',
    'pre_supplementary_motor_area_left'
]

# Create figure for visualization
plt.figure(figsize=(15, 10))

# Get a sample of LFP data (10 seconds at 400Hz = 4000 samples)
# Start from 1 minute into the recording to skip potential artifacts
start_sample = int(60 * lfps.rate)  # 1 minute
duration_samples = int(10 * lfps.rate)  # 10 seconds
time_axis = np.arange(duration_samples) / lfps.rate  # Time in seconds

# Plot LFP for each region
for i, region in enumerate(regions_to_plot):
    # Find electrode indices for this region
    region_electrodes = electrode_info[electrode_info['location'] == region].index.tolist()
    
    if region_electrodes:
        # Select one electrode from this region
        electrode_idx = np.where(electrode_indices == region_electrodes[0])[0][0]
        
        # Get LFP data for this electrode
        lfp_data = lfps.data[start_sample:start_sample+duration_samples, electrode_idx]
        
        # Plot LFP
        plt.subplot(len(regions_to_plot), 1, i+1)
        plt.plot(time_axis, lfp_data)
        plt.title(f"LFP from {region} (electrode {region_electrodes[0]})")
        plt.ylabel('Voltage (V)')
        
        if i == len(regions_to_plot) - 1:
            plt.xlabel('Time (s)')
        else:
            plt.xticks([])

plt.tight_layout()
plt.savefig('tmp_scripts/lfp_different_regions.png')

# Now create a plot to show LFP during a trial
# First, get trial information
trials = nwb.trials.to_dataframe()
print(f"\nNumber of trials: {len(trials)}")
print(f"Trial columns: {trials.columns.tolist()}")

# Select one trial to visualize
trial_idx = 5  # Arbitrary trial
trial = trials.iloc[trial_idx]

# Get key timestamps
trial_start = trial['timestamps_FixationCross']
encoding1_start = trial['timestamps_Encoding1']
encoding1_end = trial['timestamps_Encoding1_end']
encoding2_start = trial['timestamps_Encoding2']
encoding2_end = trial['timestamps_Encoding2_end']
encoding3_start = trial['timestamps_Encoding3']
encoding3_end = trial['timestamps_Encoding3_end']
maintenance_start = trial['timestamps_Maintenance']
probe_start = trial['timestamps_Probe']
response_time = trial['timestamps_Response']

# Convert timestamps to sample indices
def time_to_sample(timestamp):
    # Convert time to sample index
    return int((timestamp - lfps.starting_time) * lfps.rate)

# Get sample indices
trial_start_sample = time_to_sample(trial_start)
encoding1_start_sample = time_to_sample(encoding1_start)
encoding1_end_sample = time_to_sample(encoding1_end)
encoding2_start_sample = time_to_sample(encoding2_start)
encoding2_end_sample = time_to_sample(encoding2_end)
encoding3_start_sample = time_to_sample(encoding3_start)
encoding3_end_sample = time_to_sample(encoding3_end)
maintenance_start_sample = time_to_sample(maintenance_start)
probe_start_sample = time_to_sample(probe_start)
response_sample = time_to_sample(response_time)

# Calculate duration in samples (add buffer)
buffer_samples = int(1 * lfps.rate)  # 1 second buffer
start_sample = trial_start_sample - buffer_samples
end_sample = response_sample + buffer_samples
duration_samples = end_sample - start_sample

# Create time axis for trial
trial_time_axis = np.arange(duration_samples) / lfps.rate
trial_time_axis = trial_time_axis + (trial_start - buffer_samples / lfps.rate - lfps.starting_time)

# Create figure for trial visualization
plt.figure(figsize=(15, 12))

# Plot LFP for each region during the trial
for i, region in enumerate(regions_to_plot[:4]):  # Just plot 4 regions to keep it cleaner
    # Find electrode indices for this region
    region_electrodes = electrode_info[electrode_info['location'] == region].index.tolist()
    
    if region_electrodes:
        # Select one electrode from this region
        electrode_idx = np.where(electrode_indices == region_electrodes[0])[0][0]
        
        # Get LFP data for this electrode during the trial
        lfp_data = lfps.data[start_sample:end_sample, electrode_idx]
        
        # Plot LFP
        plt.subplot(4, 1, i+1)
        plt.plot(trial_time_axis, lfp_data)
        plt.title(f"LFP from {region} (electrode {region_electrodes[0]})")
        plt.ylabel('Voltage (V)')
        
        # Add vertical lines for event markers
        plt.axvline(x=trial_start, color='k', linestyle='--', label='Trial Start')
        plt.axvline(x=encoding1_start, color='g', linestyle='--', label='Encoding 1 Start')
        plt.axvline(x=encoding1_end, color='g', linestyle=':')
        plt.axvline(x=encoding2_start, color='b', linestyle='--', label='Encoding 2 Start')
        plt.axvline(x=encoding2_end, color='b', linestyle=':')
        plt.axvline(x=encoding3_start, color='c', linestyle='--', label='Encoding 3 Start')
        plt.axvline(x=encoding3_end, color='c', linestyle=':')
        plt.axvline(x=maintenance_start, color='m', linestyle='--', label='Maintenance Start')
        plt.axvline(x=probe_start, color='r', linestyle='--', label='Probe Start')
        plt.axvline(x=response_time, color='y', label='Response')
        
        if i == 0:
            plt.legend(loc='upper right', fontsize='small')
        
        if i == 3:  # Only add xlabel to bottom plot
            plt.xlabel('Time (s)')

plt.tight_layout()
plt.savefig('tmp_scripts/lfp_during_trial.png')

# Close the file
io.close()