"""
This script explores the basic structure of the NWB file from subject 20,
showing the main groups and datasets available for analysis.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs("tmp_scripts", exist_ok=True)

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/9fdbe18f-158f-47c5-ba67-4c56118d6cf5/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Extract basic metadata
print("NWB File Basic Metadata:")
print("-----------------------")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Experiment Description: {nwb.experiment_description}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Sex: {nwb.subject.sex}")
print(f"Subject Species: {nwb.subject.species}")
print()

# Print available acquisition data
print("Available Acquisition Data:")
print("-------------------------")
for name in nwb.acquisition:
    item = nwb.acquisition[name]
    print(f"Name: {name}, Type: {type(item).__name__}")
    if hasattr(item, 'data'):
        print(f"  Data shape: {item.data.shape}")
    if hasattr(item, 'timestamps'):
        print(f"  Timestamps shape: {item.timestamps.shape}")
print()

# Print trial information
print("Trial Information:")
print("-----------------")
trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials)}")
print("Trial columns:")
for col in trials.colnames:
    print(f"  {col}")
print()

# Print unit information
print("Neural Unit Information:")
print("-----------------------")
units = nwb.units
print(f"Number of units: {len(units)}")
print("Unit columns:")
for col in units.colnames:
    print(f"  {col}")
print()

# Print electrode information
print("Electrode Information:")
print("---------------------")
electrodes = nwb.electrodes
print(f"Number of electrodes: {len(electrodes)}")
print("Electrode columns:")
for col in electrodes.colnames:
    print(f"  {col}")
print()

# Create a histogram of the number of spikes per unit
plt.figure(figsize=(10, 6))
spike_counts = [len(units['spike_times'][i]) for i in range(len(units))]
plt.hist(spike_counts, bins=10)
plt.xlabel('Number of Spikes')
plt.ylabel('Count')
plt.title('Histogram of Spike Counts per Unit')
plt.savefig('tmp_scripts/spike_counts_histogram.png')

# Create a plot showing electrode locations (if available)
locations = electrodes['location'].data[:]
unique_locations = np.unique([str(loc) for loc in locations])

plt.figure(figsize=(10, 6))
for i, location in enumerate(unique_locations):
    # Count electrodes in each location
    count = sum(1 for loc in locations if str(loc) == location)
    plt.bar(i, count, label=location)

plt.xlabel('Location')
plt.ylabel('Number of Electrodes')
plt.title('Number of Electrodes by Brain Region')
plt.xticks(range(len(unique_locations)), unique_locations, rotation=45, ha='right')
plt.tight_layout()
plt.savefig('tmp_scripts/electrode_locations.png')

print("Plots saved to tmp_scripts directory")