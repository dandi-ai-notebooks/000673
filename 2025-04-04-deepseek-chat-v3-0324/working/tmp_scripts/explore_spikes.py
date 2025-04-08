# Script to explore spike data from NWB file
# Shows spike rasters and PSTHs aligned to stimulus events

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/000673/assets/f3a72b16-6e70-4adf-8384-28224ce212a8/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get units data
units = nwb.units
unit_ids = units["id"].data[:]  # Get all unit IDs

# Pick 3 units to plot
plot_units = unit_ids[:3] 

# Get trial information
trials = nwb.intervals["trials"]
probe_times = trials["timestamps_Probe"][:]  # Times when probe stimuli occurred
loads = trials["loads"][:]  # Memory loads for each trial (1, 2, or 3 items)

# Create figure with rasters and PSTHs
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 2])

# Plot raster
for i, unit_id in enumerate(plot_units):
    spike_times = units["spike_times"][unit_id]
    # Plot spikes around probe onset for all trials
    for t, load in zip(probe_times, loads):
        # Select spikes within 1s before and after probe
        mask = (spike_times >= t - 1) & (spike_times <= t + 1)
        rel_spikes = spike_times[mask] - t
        ax1.scatter(rel_spikes, np.ones_like(rel_spikes) * t,
                   s=1, color=f'C{i}', label=f'Unit {unit_id}')

ax1.set_title('Spike Rasters Aligned to Probe Onset')
ax1.set_xlabel('Time from probe (s)')
ax1.set_ylabel('Trial Time')
ax1.legend()

# Plot PSTH
bins = np.linspace(-1, 1, 41)
for i, unit_id in enumerate(plot_units):
    spike_times = units["spike_times"][unit_id]
    counts = np.zeros(len(bins)-1)
    for t in probe_times:
        mask = (spike_times >= t - 1) & (spike_times <= t + 1)
        rel_spikes = spike_times[mask] - t
        h, _ = np.histogram(rel_spikes, bins=bins)
        counts += h
    
    ax2.bar(bins[:-1], counts/len(probe_times), width=np.diff(bins),
            align='edge', alpha=0.5, color=f'C{i}', label=f'Unit {unit_id}')

ax2.set_title('Peri-Stimulus Time Histogram (PSTH)')
ax2.set_xlabel('Time from probe (s)')
ax2.set_ylabel('Spikes per trial')
ax2.legend()

plt.tight_layout()
plt.savefig('tmp_scripts/spike_analysis.png')
plt.close()