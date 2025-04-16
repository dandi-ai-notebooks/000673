"""
This script explores the trial information and behavioral data from the Sternberg task.
We'll examine the structure of trials, task parameters, and subject performance.
"""

import numpy as np
import matplotlib.pyplot as plt
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

# Get trials data
print("Extracting trials data...")
trials = nwb.intervals['trials']
trials_df = trials.to_dataframe()

# Print basic statistics
print(f"Number of trials: {len(trials_df)}")
print(f"Columns in trials data: {trials_df.columns.tolist()}")

# Distribution of memory loads
print("\nMemory load distribution:")
load_counts = trials_df['loads'].value_counts()
print(load_counts)

# Calculate accuracy by memory load
print("\nAccuracy by memory load:")
accuracy_by_load = trials_df.groupby('loads')['response_accuracy'].mean() * 100
print(accuracy_by_load)

# Calculate reaction time by memory load
trials_df['reaction_time'] = trials_df['timestamps_Response'] - trials_df['timestamps_Probe']
print("\nMean reaction time (seconds) by memory load:")
rt_by_load = trials_df.groupby('loads')['reaction_time'].mean()
print(rt_by_load)

# Plot accuracy by memory load
plt.figure(figsize=(10, 6))
accuracy_by_load.plot(kind='bar')
plt.xlabel('Memory Load')
plt.ylabel('Accuracy (%)')
plt.title('Response Accuracy by Memory Load')
plt.savefig('explore/accuracy_by_load.png', dpi=300, bbox_inches='tight')

# Plot reaction time by memory load
plt.figure(figsize=(10, 6))
rt_by_load.plot(kind='bar')
plt.xlabel('Memory Load')
plt.ylabel('Reaction Time (seconds)')
plt.title('Reaction Time by Memory Load')
plt.savefig('explore/rt_by_load.png', dpi=300, bbox_inches='tight')

# Plot distribution of memory loads
plt.figure(figsize=(10, 6))
load_counts.plot(kind='bar')
plt.xlabel('Memory Load')
plt.ylabel('Count')
plt.title('Distribution of Memory Loads')
plt.savefig('explore/load_distribution.png', dpi=300, bbox_inches='tight')

# Calculate performance based on whether probe was in memory or not
print("\nAccuracy by probe type:")
in_out_accuracy = trials_df.groupby('probe_in_out')['response_accuracy'].mean() * 100
print(f"Probe in memory: {in_out_accuracy.get(1, 'N/A')}%")
print(f"Probe not in memory: {in_out_accuracy.get(0, 'N/A')}%")

# Plot accuracy by probe type
plt.figure(figsize=(10, 6))
labels = ['Not in memory', 'In memory']
probe_types = [0, 1]
accuracies = [in_out_accuracy.get(pt, 0) for pt in probe_types]
plt.bar(labels, accuracies)
plt.xlabel('Probe Type')
plt.ylabel('Accuracy (%)')
plt.title('Response Accuracy by Probe Type')
plt.savefig('explore/accuracy_by_probe.png', dpi=300, bbox_inches='tight')

# Analyze trial duration
trials_df['trial_duration'] = trials_df['stop_time'] - trials_df['start_time']
trials_df['encoding_duration'] = trials_df['timestamps_Encoding1_end'] - trials_df['timestamps_Encoding1']
trials_df['maintenance_duration'] = trials_df['timestamps_Probe'] - trials_df['timestamps_Maintenance']

# Calculate mean durations
print("\nMean trial phase durations (seconds):")
print(f"Total trial: {trials_df['trial_duration'].mean()}")
print(f"Encoding (first item): {trials_df['encoding_duration'].mean()}")
print(f"Maintenance: {trials_df['maintenance_duration'].mean()}")

print("Plots saved to 'explore' directory")