"""
This script explores the neural units (single neurons) in Dandiset 000673 (sub-10).
The goal is to understand the properties of recorded neurons and their activity patterns.
"""

import numpy as np
import pandas as pd
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Load
url = "https://api.dandiarchive.org/api/assets/c03df798-01fc-4023-ab51-e1721e2db93c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get units information
units_df = nwb.units.to_dataframe()
print(f"Total number of units: {len(units_df)}")
print(f"\nUnit columns: {list(units_df.columns)}")

# Get electrode information
electrodes_df = nwb.electrodes.to_dataframe()

# Create a simpler script that doesn't depend on electrode-unit mapping
# since we're having issues with the electrodes column
print("\nElectrode locations:")
print(electrodes_df['location'].value_counts())

# Since we don't have reliable unit-to-region mapping from the dataframe,
# we'll just look at the overall unit activity

# Plot spike counts for all units
spike_counts = []
unit_ids = []
for i, unit_id in enumerate(nwb.units.id.data):
    spike_times = nwb.units['spike_times'][i]
    spike_counts.append(len(spike_times))
    unit_ids.append(unit_id)

plt.figure(figsize=(12, 6))
plt.bar(range(len(spike_counts)), spike_counts)
plt.title('Number of Spikes per Unit')
plt.xlabel('Unit Index')
plt.ylabel('Number of Spikes')
plt.savefig('explore/spike_counts.png')

# Get trials data
trials_df = nwb.trials.to_dataframe()

# Select a unit to analyze with respect to trials
unit_idx = 0  # Choose the first unit
unit_id = units_df.index[unit_idx]
spike_times = nwb.units['spike_times'][unit_idx]
print(f"\nAnalyzing Unit {unit_id}")
print(f"Number of spikes: {len(spike_times)}")

# Create a raster plot for this unit
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Prepare trial data for plotting
trial_starts = trials_df['start_time'].values
trial_ends = trials_df['stop_time'].values
trial_loads = trials_df['loads'].values
fixation_times = trials_df['timestamps_FixationCross'].values
probe_times = trials_df['timestamps_Probe'].values
response_times = trials_df['timestamps_Response'].values

# Top plot: Spike raster aligned to trial onsets
ax1 = plt.subplot(gs[0])

# For each trial
for i in range(len(trials_df)):
    # Get the trial load (1 or 3)
    load = trial_loads[i]
    color = 'blue' if load == 1 else 'red'
    
    # Calculate the spikes relative to trial start
    trial_start = trial_starts[i]
    trial_end = trial_ends[i]
    
    # Find spikes within this trial
    trial_spikes = [t for t in spike_times if trial_start <= t <= trial_end]
    
    # Plot spikes for this trial
    if trial_spikes:
        ax1.vlines([t - trial_start for t in trial_spikes], i - 0.4, i + 0.4, 
                   color=color, linewidth=0.5)

# Add markers for different task events (using trial 0 as reference for timing)
avg_fixation_time = np.mean(fixation_times - trial_starts)
avg_probe_time = np.mean(probe_times - trial_starts)
avg_response_time = np.mean(response_times - trial_starts)

ax1.axvline(x=avg_fixation_time, color='green', linestyle='--', alpha=0.7, label='Avg Fixation')
ax1.axvline(x=avg_probe_time, color='purple', linestyle='--', alpha=0.7, label='Avg Probe')
ax1.axvline(x=avg_response_time, color='orange', linestyle='--', alpha=0.7, label='Avg Response')

ax1.set_xlim(-0.5, 20)  # Adjust as needed
ax1.set_xlabel('Time from Trial Start (s)')
ax1.set_ylabel('Trial Number')
ax1.set_title(f'Unit {unit_id} Spikes by Trial')
ax1.legend()

# Bottom plot: Average firing rate
ax2 = plt.subplot(gs[1], sharex=ax1)

# Calculate trial-averaged PSTH
bin_size = 0.1  # seconds
max_trial_duration = 20  # seconds (adjust as needed)
bins = np.arange(0, max_trial_duration + bin_size, bin_size)
psth = np.zeros(len(bins) - 1)
trial_count = 0

for i in range(len(trials_df)):
    trial_start = trial_starts[i]
    trial_end = min(trial_ends[i], trial_start + max_trial_duration)  # Cap at max duration
    
    # Find spikes within this trial
    trial_spikes = np.array([t - trial_start for t in spike_times if trial_start <= t <= trial_end])
    
    if len(trial_spikes) > 0:
        hist, _ = np.histogram(trial_spikes, bins=bins)
        psth += hist
        trial_count += 1

if trial_count > 0:
    psth = psth / (trial_count * bin_size)  # Convert to firing rate (Hz)

ax2.bar(bins[:-1], psth, width=bin_size, alpha=0.7)
ax2.set_xlabel('Time from Trial Start (s)')
ax2.set_ylabel('Firing Rate (Hz)')
ax2.set_title(f'Average Firing Rate')
ax2.axvline(x=avg_fixation_time, color='green', linestyle='--', alpha=0.7)
ax2.axvline(x=avg_probe_time, color='purple', linestyle='--', alpha=0.7)
ax2.axvline(x=avg_response_time, color='orange', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('explore/unit_raster_psth.png')

# Now analyze firing rate during different task periods
# We'll look at the encoding, maintenance, and probe periods

# Create bins for each period across all trials
period_names = ['Fixation', 'Encoding', 'Maintenance', 'Probe', 'Response']
period_rates = {name: [] for name in period_names}

# For all units
all_units_rates = []

for unit_idx in range(len(units_df)):
    unit_id = units_df.index[unit_idx]
    spike_times = nwb.units['spike_times'][unit_idx]
    unit_rates = []
    
    for trial_idx in range(len(trials_df)):
        # Fixation period
        start_time = fixation_times[trial_idx]
        if trial_loads[trial_idx] == 1:
            # For load 1, encoding is just Encoding1
            encoding_start = trials_df['timestamps_Encoding1'].iloc[trial_idx]
            encoding_end = trials_df['timestamps_Encoding1_end'].iloc[trial_idx]
        else:
            # For load 3, encoding spans Encoding1 to Encoding3_end
            encoding_start = trials_df['timestamps_Encoding1'].iloc[trial_idx]
            encoding_end = trials_df['timestamps_Encoding3_end'].iloc[trial_idx]
        
        maintenance_start = trials_df['timestamps_Maintenance'].iloc[trial_idx]
        probe_start = trials_df['timestamps_Probe'].iloc[trial_idx]
        response_time = trials_df['timestamps_Response'].iloc[trial_idx]
        
        # Define all periods
        periods = {
            'Fixation': (start_time, encoding_start),
            'Encoding': (encoding_start, encoding_end),
            'Maintenance': (maintenance_start, probe_start),
            'Probe': (probe_start, response_time),
            'Response': (response_time, response_time + 0.5)  # 500ms after response
        }
        
        # Calculate firing rates for each period
        for name, (period_start, period_end) in periods.items():
            # Count spikes in this period
            period_spikes = [t for t in spike_times if period_start <= t <= period_end]
            duration = period_end - period_start
            if duration > 0:
                rate = len(period_spikes) / duration  # Hz
                period_rates[name].append(rate)
    
    # Calculate average rates across all trials for this unit
    for name in period_names:
        if period_rates[name]:
            avg_rate = np.mean(period_rates[name])
            unit_rates.append(avg_rate)
        else:
            unit_rates.append(0)
    
    all_units_rates.append(unit_rates)
    
    # Reset for next unit
    period_rates = {name: [] for name in period_names}

# Convert to numpy array
all_units_rates = np.array(all_units_rates)

# Plot average firing rate for each task period across units
plt.figure(figsize=(10, 6))
means = np.mean(all_units_rates, axis=0)
std_errs = np.std(all_units_rates, axis=0) / np.sqrt(len(all_units_rates))

plt.bar(range(len(period_names)), means, yerr=std_errs, alpha=0.7)
plt.xticks(range(len(period_names)), period_names)
plt.ylabel('Mean Firing Rate (Hz)')
plt.title('Average Firing Rate Across Task Periods (All Units)')
plt.savefig('explore/firing_rate_by_period.png')

# Close file
h5_file.close()