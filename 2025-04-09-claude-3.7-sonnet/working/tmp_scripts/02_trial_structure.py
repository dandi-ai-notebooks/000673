"""
This script explores the trial structure of the Sternberg working memory task
in Dandiset 000673, focusing on the task phases, memory loads, and behavioral performance.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style for plots
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Get trials data as DataFrame
trials_df = nwb.trials.to_dataframe()

# Basic trial information
print("=== TRIAL INFORMATION ===")
print(f"Number of trials: {len(trials_df)}")
print(f"Trial columns: {trials_df.columns.tolist()}")

# Check memory loads
loads = trials_df['loads'].astype(int)
load_counts = loads.value_counts().sort_index()
print(f"\nMemory loads: {load_counts.index.tolist()}")
print(f"Count per load: {load_counts.values.tolist()}")

# Calculate accuracy by memory load
accuracy_by_load = trials_df.groupby('loads')['response_accuracy'].mean() * 100
print("\nAccuracy by memory load:")
for load, acc in accuracy_by_load.items():
    print(f"  Load {int(load)}: {acc:.1f}%")

# Plot accuracy by memory load
plt.figure(figsize=(10, 6))
plt.bar(accuracy_by_load.index.astype(int), accuracy_by_load.values)
plt.xlabel('Memory Load')
plt.ylabel('Accuracy (%)')
plt.title('Response Accuracy by Memory Load')
plt.xticks(accuracy_by_load.index.astype(int))
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tmp_scripts/accuracy_by_load.png', dpi=300)

# Calculate trial durations
trials_df['encoding_duration'] = (
    trials_df['timestamps_Encoding3_end'] - trials_df['timestamps_Encoding1']
)
trials_df['maintenance_duration'] = (
    trials_df['timestamps_Probe'] - trials_df['timestamps_Encoding3_end']
)
trials_df['response_time'] = (
    trials_df['timestamps_Response'] - trials_df['timestamps_Probe']
)
trials_df['total_duration'] = (
    trials_df['stop_time'] - trials_df['start_time']
)

# Calculate average durations
print("\nAverage trial durations (seconds):")
print(f"  Encoding phase: {trials_df['encoding_duration'].mean():.2f}")
print(f"  Maintenance phase: {trials_df['maintenance_duration'].mean():.2f}")
print(f"  Response time: {trials_df['response_time'].mean():.2f}")
print(f"  Total trial: {trials_df['total_duration'].mean():.2f}")

# Visualize trial structure timeline
plt.figure(figsize=(12, 8))

# Sample a few trials for visualization
sample_trials = trials_df.iloc[0:5]

# Plot timeline for sample trials
for i, (idx, trial) in enumerate(sample_trials.iterrows()):
    # Calculate start times relative to trial start
    rel_fixation = trial['timestamps_FixationCross'] - trial['start_time']
    rel_enc1 = trial['timestamps_Encoding1'] - trial['start_time']
    rel_enc1_end = trial['timestamps_Encoding1_end'] - trial['start_time']
    rel_enc2 = trial['timestamps_Encoding2'] - trial['start_time'] if not np.isnan(trial['timestamps_Encoding2']) else None
    rel_enc2_end = trial['timestamps_Encoding2_end'] - trial['start_time'] if not np.isnan(trial['timestamps_Encoding2_end']) else None
    rel_enc3 = trial['timestamps_Encoding3'] - trial['start_time'] if not np.isnan(trial['timestamps_Encoding3']) else None
    rel_enc3_end = trial['timestamps_Encoding3_end'] - trial['start_time'] if not np.isnan(trial['timestamps_Encoding3_end']) else None
    rel_maint = trial['timestamps_Maintenance'] - trial['start_time']
    rel_probe = trial['timestamps_Probe'] - trial['start_time']
    rel_resp = trial['timestamps_Response'] - trial['start_time']
    
    # Plot trial phases
    plt.plot([rel_fixation, rel_enc1], [i, i], 'k-', linewidth=2, alpha=0.5)
    plt.axvline(x=rel_fixation, color='gray', linestyle='--', alpha=0.5)
    
    # Encoding 1
    plt.plot([rel_enc1, rel_enc1_end], [i, i], 'r-', linewidth=4, label='Encoding' if i == 0 else "")
    
    # Encoding 2 (if present)
    if rel_enc2 is not None and rel_enc2_end is not None:
        plt.plot([rel_enc2, rel_enc2_end], [i, i], 'r-', linewidth=4)
    
    # Encoding 3 (if present)
    if rel_enc3 is not None and rel_enc3_end is not None:
        plt.plot([rel_enc3, rel_enc3_end], [i, i], 'r-', linewidth=4)
    
    # Maintenance
    plt.plot([rel_maint, rel_probe], [i, i], 'b-', linewidth=4, label='Maintenance' if i == 0 else "")
    
    # Probe and response
    plt.plot([rel_probe, rel_resp], [i, i], 'g-', linewidth=4, label='Probe & Response' if i == 0 else "")
    
    # Add markers for key events
    plt.scatter(rel_fixation, i, color='black', s=50, label='Fixation' if i == 0 else "")
    plt.scatter(rel_enc1, i, color='red', s=50)
    if rel_enc2 is not None:
        plt.scatter(rel_enc2, i, color='red', s=50)
    if rel_enc3 is not None:
        plt.scatter(rel_enc3, i, color='red', s=50)
    plt.scatter(rel_maint, i, color='blue', s=50)
    plt.scatter(rel_probe, i, color='green', s=50)
    plt.scatter(rel_resp, i, color='purple', s=50, label='Response' if i == 0 else "")

# Add load information to y-axis labels
trial_labels = [f"Trial {i+1} (Load {int(trial['loads'])})" for i, (_, trial) in enumerate(sample_trials.iterrows())]
plt.yticks(range(len(sample_trials)), trial_labels)

plt.xlabel('Time (s)')
plt.title('Trial Structure Timeline (First 5 Trials)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('tmp_scripts/trial_timeline.png', dpi=300)

# Explore in/out probes
if 'probe_in_out' in trials_df.columns:
    in_out_counts = trials_df['probe_in_out'].value_counts()
    print("\nProbe types:")
    print(f"  In-memory-set probes: {in_out_counts.get(1, 0)}")
    print(f"  Out-of-memory-set probes: {in_out_counts.get(0, 0)}")
    
    # Accuracy by probe type
    accuracy_by_probe_type = trials_df.groupby('probe_in_out')['response_accuracy'].mean() * 100
    print("\nAccuracy by probe type:")
    for probe_type, acc in accuracy_by_probe_type.items():
        type_name = "In memory set" if probe_type == 1 else "Out of memory set"
        print(f"  {type_name}: {acc:.1f}%")

# Plot response times by memory load
plt.figure(figsize=(10, 6))
sns.boxplot(x=trials_df['loads'].astype(int), y=trials_df['response_time'])
plt.xlabel('Memory Load')
plt.ylabel('Response Time (s)')
plt.title('Response Time by Memory Load')
plt.tight_layout()
plt.savefig('tmp_scripts/response_time.png', dpi=300)

# Plot response times by accuracy
plt.figure(figsize=(10, 6))
sns.boxplot(x=trials_df['response_accuracy'].astype(int), y=trials_df['response_time'])
plt.xlabel('Response Accuracy (0=Incorrect, 1=Correct)')
plt.ylabel('Response Time (s)')
plt.title('Response Time by Accuracy')
plt.tight_layout()
plt.savefig('tmp_scripts/response_time_by_accuracy.png', dpi=300)

# Close the file
io.close()
f.close()
file.close()