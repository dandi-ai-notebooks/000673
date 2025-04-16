"""
This script explores the trial structure and behavioral data in Dandiset 000673 (sub-10).
The goal is to understand the Sternberg task trials, accuracy, and load-dependent behavior.
"""

import numpy as np
import pandas as pd
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt

# Load
url = "https://api.dandiarchive.org/api/assets/c03df798-01fc-4023-ab51-e1721e2db93c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trials as a pandas dataframe
trials_df = nwb.trials.to_dataframe()

print(f"Total number of trials: {len(trials_df)}")
print(f"\nTrial columns: {list(trials_df.columns)}")
print(f"\nUnique memory loads: {sorted(trials_df['loads'].unique())}")

# Compute performance by memory load
performance_by_load = trials_df.groupby('loads')['response_accuracy'].mean() * 100

print("\nPerformance by memory load:")
for load, accuracy in performance_by_load.items():
    print(f"Load {load}: {accuracy:.2f}% correct")

# Plot performance by memory load
plt.figure(figsize=(10, 6))
performance_by_load.plot(kind='bar', color='skyblue')
plt.title('Task Performance by Memory Load')
plt.xlabel('Memory Load')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('explore/performance_by_load.png')

# Analyze reaction time by memory load
trials_df['reaction_time'] = trials_df['timestamps_Response'] - trials_df['timestamps_Probe']
reaction_time_by_load = trials_df.groupby('loads')['reaction_time'].mean()

print("\nMean reaction time by memory load:")
for load, rt in reaction_time_by_load.items():
    print(f"Load {load}: {rt*1000:.2f} ms")

# Plot reaction time by memory load
plt.figure(figsize=(10, 6))
reaction_time_by_load.plot(kind='bar', color='salmon')
plt.title('Reaction Time by Memory Load')
plt.xlabel('Memory Load')
plt.ylabel('Reaction Time (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('explore/reaction_time_by_load.png')

# Analysis of probe_in_out (in-memory vs not-in-memory probe)
in_vs_out_accuracy = trials_df.groupby('probe_in_out')['response_accuracy'].mean() * 100
in_vs_out_size = trials_df.groupby('probe_in_out').size()

print("\nPerformance by probe type:")
for probe_type, accuracy in in_vs_out_accuracy.items():
    type_name = "In memory" if probe_type == 1 else "Not in memory"
    count = in_vs_out_size[probe_type]
    print(f"{type_name} ({count} trials): {accuracy:.2f}% correct")

# Plot pie chart of trial distribution
plt.figure(figsize=(8, 8))
labels = ['In memory', 'Not in memory']
sizes = [in_vs_out_size[1], in_vs_out_size[0]]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.axis('equal')
plt.title('Distribution of Trial Types')
plt.savefig('explore/trial_distribution.png')

# Save the trials dataframe for first 5 rows to examine structure
print("\nFirst 5 trials:")
print(trials_df.head(5).to_string())

# Close file
h5_file.close()