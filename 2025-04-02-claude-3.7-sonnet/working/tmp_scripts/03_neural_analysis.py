"""
This script analyzes neural activity patterns in relation to the Sternberg working memory task,
focusing on spike timing, firing rates during different task phases, and responses to memory load.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Ensure the output directory exists
os.makedirs("tmp_scripts", exist_ok=True)

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/9fdbe18f-158f-47c5-ba67-4c56118d6cf5/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Access the trials and units data
trials = nwb.intervals["trials"]
units = nwb.units
electrodes = nwb.electrodes

print(f"Analyzing {len(units)} units across {len(trials)} trials")

# Get unit and electrode information
unit_ids = units.id.data[:]
electrode_indices = units["electrodes"].data[:]
electrode_locations = [electrodes["location"][idx] for idx in electrode_indices]

# Function to create raster plot for a single unit
def create_raster_plot(unit_index, event_name, window=(-0.5, 1.5), bin_size=0.05):
    """Create a raster plot and PSTH for a unit around a specific event"""
    unit_spike_times = units["spike_times"][unit_index]
    
    # Get event times
    if event_name == "probe":
        event_times = trials["timestamps_Probe"][:]
    elif event_name == "maintenance":
        event_times = trials["timestamps_Maintenance"][:]
    else:
        event_times = trials["timestamps_FixationCross"][:]
    
    # For raster plot - collect spike times relative to each event
    trial_spikes = []
    for event_time in event_times:
        # Get spikes in window relative to event
        mask = (unit_spike_times >= event_time + window[0]) & (unit_spike_times <= event_time + window[1])
        relative_spikes = unit_spike_times[mask] - event_time
        trial_spikes.append(relative_spikes)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Raster plot
    for i, spikes in enumerate(trial_spikes):
        ax1.plot(spikes, np.ones_like(spikes) * i, '|', color='k', markersize=5)
    
    ax1.set_ylabel('Trial')
    ax1.set_title(f'Unit {unit_ids[unit_index]} responses to {event_name} onset')
    
    # PSTH
    bins = np.arange(window[0], window[1], bin_size)
    all_spikes = np.concatenate(trial_spikes)
    counts, edges = np.histogram(all_spikes, bins=bins)
    rate = counts / (len(event_times) * bin_size)  # Hz
    centers = (edges[:-1] + edges[1:]) / 2
    
    ax2.bar(centers, rate, width=bin_size*0.8, alpha=0.6)
    ax2.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time from event onset (s)')
    ax2.set_ylabel('Firing rate (Hz)')
    
    plt.tight_layout()
    return fig

# Choose a unit that has a good number of spikes for visualization
spike_counts = [len(units['spike_times'][i]) for i in range(len(units))]
unit_indices = np.argsort(spike_counts)[-5:]  # Get indices of 5 most active units
selected_unit = unit_indices[0]  # Select the most active unit

# Create raster plots for three event types
raster_probe = create_raster_plot(selected_unit, "probe")
raster_probe.savefig('tmp_scripts/raster_probe.png')

raster_maint = create_raster_plot(selected_unit, "maintenance")
raster_maint.savefig('tmp_scripts/raster_maintenance.png')

raster_fix = create_raster_plot(selected_unit, "fixation")
raster_fix.savefig('tmp_scripts/raster_fixation.png')

# Analyze firing rates during different task phases
print("\nFiring rate analysis for selected unit:")
print(f"Unit ID: {unit_ids[selected_unit]}")
print(f"Electrode location: {electrode_locations[selected_unit]}")

# Function to calculate mean firing rate in a window around events
def calc_firing_rate(spike_times, event_times, window=(-0.5, 0.5)):
    """Calculate mean firing rate in a window around events"""
    total_spikes = 0
    for event_time in event_times:
        mask = (spike_times >= event_time + window[0]) & (spike_times <= event_time + window[1])
        total_spikes += np.sum(mask)
    
    # Calculate rate in Hz
    total_duration = len(event_times) * (window[1] - window[0])
    return total_spikes / total_duration if total_duration > 0 else 0

# Get different event times
fixation_times = trials["timestamps_FixationCross"][:]
encoding1_times = trials["timestamps_Encoding1"][:]
encoding3_times = trials["timestamps_Encoding3"][:]
maintenance_times = trials["timestamps_Maintenance"][:]
probe_times = trials["timestamps_Probe"][:]

# Get memory load conditions
loads = trials["loads"][:]
load1_mask = loads == 1
load3_mask = loads == 3

probe_times_load1 = probe_times[load1_mask]
probe_times_load3 = probe_times[load3_mask]
maintenance_times_load1 = maintenance_times[load1_mask]
maintenance_times_load3 = maintenance_times[load3_mask]

# Calculate firing rates for different task phases and conditions
unit_spikes = units["spike_times"][selected_unit]
baseline_rate = calc_firing_rate(unit_spikes, fixation_times)
encoding1_rate = calc_firing_rate(unit_spikes, encoding1_times)
encoding3_rate = calc_firing_rate(unit_spikes, encoding3_times)
maintenance_rate = calc_firing_rate(unit_spikes, maintenance_times)
probe_rate = calc_firing_rate(unit_spikes, probe_times)

# Load comparison
maintenance_rate_load1 = calc_firing_rate(unit_spikes, maintenance_times_load1)
maintenance_rate_load3 = calc_firing_rate(unit_spikes, maintenance_times_load3)
probe_rate_load1 = calc_firing_rate(unit_spikes, probe_times_load1)
probe_rate_load3 = calc_firing_rate(unit_spikes, probe_times_load3)

print(f"\nMean firing rates (Hz):")
print(f"  Baseline (fixation): {baseline_rate:.2f}")
print(f"  Encoding (first item): {encoding1_rate:.2f}")
print(f"  Encoding (third item): {encoding3_rate:.2f}")
print(f"  Maintenance: {maintenance_rate:.2f}")
print(f"  Probe: {probe_rate:.2f}")

print(f"\nMemory load comparison (Hz):")
print(f"  Maintenance (Load 1): {maintenance_rate_load1:.2f}")
print(f"  Maintenance (Load 3): {maintenance_rate_load3:.2f}")
print(f"  Probe (Load 1): {probe_rate_load1:.2f}")
print(f"  Probe (Load 3): {probe_rate_load3:.2f}")

# Plot firing rates by task phase
phases = ['Baseline', 'Encoding1', 'Encoding3', 'Maintenance', 'Probe']
rates = [baseline_rate, encoding1_rate, encoding3_rate, maintenance_rate, probe_rate]

plt.figure(figsize=(12, 6))
plt.bar(phases, rates)
plt.ylabel('Firing Rate (Hz)')
plt.title(f'Unit {unit_ids[selected_unit]} - Firing Rate by Task Phase')
plt.savefig('tmp_scripts/firing_rate_by_phase.png')

# Plot load comparison 
plt.figure(figsize=(10, 6))
phases = ['Maintenance\nLoad 1', 'Maintenance\nLoad 3', 'Probe\nLoad 1', 'Probe\nLoad 3']
rates = [maintenance_rate_load1, maintenance_rate_load3, probe_rate_load1, probe_rate_load3]
plt.bar(phases, rates)
plt.ylabel('Firing Rate (Hz)')
plt.title(f'Unit {unit_ids[selected_unit]} - Firing Rate by Memory Load')
plt.savefig('tmp_scripts/firing_rate_by_load.png')

print("\nPlots saved to tmp_scripts directory")