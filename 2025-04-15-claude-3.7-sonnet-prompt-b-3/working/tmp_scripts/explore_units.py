# This script explores the neural units (single neurons) in the dataset
# focusing on spike times, electrode locations, and activity patterns

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode and unit information
electrode_df = nwb.electrodes.to_dataframe()
units_df = nwb.units.to_dataframe()

print(f"Total number of units: {len(units_df)}")

# Get locations for each unit
unit_locations = []
for idx, unit in units_df.iterrows():
    electrode_id = unit['electrodes']
    if electrode_id is not None:  # Some units might not have electrode info
        location = electrode_df.loc[electrode_id, 'location']
        unit_locations.append(location)
    else:
        unit_locations.append('Unknown')

units_df['location'] = unit_locations

# Count units per brain region
region_counts = pd.Series(unit_locations).value_counts()
print("\nNumber of units per brain region:")
for region, count in region_counts.items():
    print(f"  {region}: {count} units")

# Create plot for units per brain region
plt.figure(figsize=(12, 6))
sns.barplot(x=region_counts.index, y=region_counts.values)
plt.title('Number of Units per Brain Region')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('tmp_scripts/units_per_region.png')

# Examine spike times
print("\nExamining spike times...")

# Get trial information for alignment
trials = nwb.trials.to_dataframe()

# Random pick 5 units for detailed analysis
sample_units = np.random.choice(units_df.index, size=5, replace=False)

# Create plot to show spike patterns for sample units
plt.figure(figsize=(15, 10))

# Get a sample of trials
sample_trials = np.random.choice(trials.index, size=10, replace=False)
sample_trials = sorted(sample_trials)

# Create a raster plot for sample units across sample trials
plt.figure(figsize=(15, 12))

# For each sampled unit
for i, unit_id in enumerate(sample_units):
    unit = units_df.loc[unit_id]
    spike_times = unit['spike_times']
    location = unit['location']
    
    # Create subplot for this unit
    ax = plt.subplot(len(sample_units), 1, i+1)
    
    # Plot spike times for each trial
    for j, trial_idx in enumerate(sample_trials):
        trial = trials.iloc[trial_idx]
        
        # Get trial start and end times
        trial_start = trial['start_time']
        trial_end = trial['stop_time']
        
        # Get spikes during this trial
        trial_spikes = spike_times[(spike_times >= trial_start) & (spike_times <= trial_end)]
        
        # Normalize spike times relative to trial start
        normalized_spikes = trial_spikes - trial_start
        
        # Plot spikes
        plt.scatter(normalized_spikes, np.ones_like(normalized_spikes) * j, 
                   color='k', marker='|', s=50)
    
    # Get key trial events (use first trial for reference)
    ref_trial = trials.iloc[sample_trials[0]]
    encoding1_time = ref_trial['timestamps_Encoding1'] - ref_trial['start_time']
    encoding2_time = ref_trial['timestamps_Encoding2'] - ref_trial['start_time']
    encoding3_time = ref_trial['timestamps_Encoding3'] - ref_trial['start_time']
    maintenance_time = ref_trial['timestamps_Maintenance'] - ref_trial['start_time']
    probe_time = ref_trial['timestamps_Probe'] - ref_trial['start_time']
    
    # Add vertical lines for key events
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Trial Start')
    plt.axvline(x=encoding1_time, color='g', linestyle='--', alpha=0.5, label='Encoding 1')
    plt.axvline(x=encoding2_time, color='b', linestyle='--', alpha=0.5, label='Encoding 2')
    plt.axvline(x=encoding3_time, color='c', linestyle='--', alpha=0.5, label='Encoding 3')
    plt.axvline(x=maintenance_time, color='m', linestyle='--', alpha=0.5, label='Maintenance')
    plt.axvline(x=probe_time, color='r', linestyle='--', alpha=0.5, label='Probe')
    
    plt.title(f'Unit {unit_id} from {location}')
    plt.ylabel('Trial')
    
    if i == 0:
        plt.legend(loc='upper right')
    
    if i == len(sample_units) - 1:
        plt.xlabel('Time from trial start (s)')
    else:
        plt.xticks([])

plt.tight_layout()
plt.savefig('tmp_scripts/unit_raster_plots.png')

# Create plot to show mean firing rates across all units during trial phases
# Calculate mean firing rate for each unit during each trial phase

# Define phases
phases = [
    ('Fixation', 'timestamps_FixationCross', 'timestamps_Encoding1'),
    ('Encoding 1', 'timestamps_Encoding1', 'timestamps_Encoding1_end'),
    ('Encoding 2', 'timestamps_Encoding2', 'timestamps_Encoding2_end'),
    ('Encoding 3', 'timestamps_Encoding3', 'timestamps_Encoding3_end'),
    ('Maintenance', 'timestamps_Maintenance', 'timestamps_Probe'),
    ('Probe', 'timestamps_Probe', 'timestamps_Response')
]

# Filter out trials with any missing phase timestamps
valid_trials = trials.dropna(subset=[col for _, col, _ in phases] + [col for _, _, col in phases])

# Get firing rates for all units
all_firing_rates = []

for unit_id in units_df.index:
    unit = units_df.loc[unit_id]
    spike_times = unit['spike_times']
    location = unit['location']
    
    unit_rates = {'unit_id': unit_id, 'location': location}
    
    # Calculate firing rate for each phase
    for phase_name, start_col, end_col in phases:
        spike_counts = []
        durations = []
        
        for _, trial in valid_trials.iterrows():
            phase_start = trial[start_col]
            phase_end = trial[end_col]
            duration = phase_end - phase_start
            
            # Count spikes in this phase
            n_spikes = np.sum((spike_times >= phase_start) & (spike_times <= phase_end))
            
            spike_counts.append(n_spikes)
            durations.append(duration)
        
        # Calculate mean rate
        if sum(durations) > 0:
            rate = sum(spike_counts) / sum(durations)
        else:
            rate = 0
            
        unit_rates[phase_name] = rate
    
    all_firing_rates.append(unit_rates)

# Convert to DataFrame
firing_rates_df = pd.DataFrame(all_firing_rates)

# Melt for easier plotting
melted_rates = pd.melt(
    firing_rates_df, 
    id_vars=['unit_id', 'location'], 
    value_vars=[p[0] for p in phases],
    var_name='Phase', 
    value_name='Firing Rate (Hz)'
)

# Plot firing rates by brain region and phase
plt.figure(figsize=(15, 10))
sns.boxplot(x='Phase', y='Firing Rate (Hz)', hue='location', data=melted_rates)
plt.title('Firing Rates by Brain Region and Trial Phase')
plt.xticks(rotation=45)
plt.legend(title='Brain Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('tmp_scripts/firing_rates_by_region_phase.png')

# Calculate mean waveforms and SNR stats
print("\nWaveform statistics:")
print(f"Mean SNR range: {units_df['waveforms_mean_snr'].min():.2f} - {units_df['waveforms_mean_snr'].max():.2f}")
print(f"Mean isolation distance: {units_df['waveforms_isolation_distance'].mean():.2f}")

# Close the file
io.close()