"""
This script analyzes the behavioral data from the Sternberg working memory task,
focusing on trial information, memory loads, and performance metrics.
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

# Access the trials data
trials = nwb.intervals["trials"]

# Print basic information about the trials
print("Behavioral Task Analysis:")
print("------------------------")
print(f"Total number of trials: {len(trials)}")

# Extract key behavioral measures
loads = trials["loads"][:]
accuracy = trials["response_accuracy"][:]
probe_in_out = trials["probe_in_out"][:]

# Calculate and print summary statistics
print("\nMemory Load Distribution:")
unique_loads, load_counts = np.unique(loads, return_counts=True)
for load, count in zip(unique_loads, load_counts):
    print(f"Load {load}: {count} trials ({count/len(loads)*100:.1f}%)")

print("\nOverall Performance:")
print(f"Accuracy: {np.mean(accuracy)*100:.1f}%")

# Calculate accuracy by memory load
print("\nAccuracy by Memory Load:")
for load in unique_loads:
    load_mask = loads == load
    load_accuracy = np.mean(accuracy[load_mask]) * 100
    print(f"Load {load}: {load_accuracy:.1f}%")

# Calculate accuracy by probe condition (in vs. out)
in_mask = probe_in_out == 1
out_mask = probe_in_out == 0
print("\nAccuracy by Probe Condition:")
print(f"In-memory probes: {np.mean(accuracy[in_mask])*100:.1f}%")
print(f"Out-of-memory probes: {np.mean(accuracy[out_mask])*100:.1f}%")

# Calculate response times
response_times = trials["timestamps_Response"][:] - trials["timestamps_Probe"][:]
print("\nResponse Time Analysis:")
print(f"Mean response time: {np.mean(response_times)*1000:.1f} ms")
print(f"Median response time: {np.median(response_times)*1000:.1f} ms")

# Plot accuracy by memory load
plt.figure(figsize=(10, 6))
load_accuracies = [np.mean(accuracy[loads == load]) * 100 for load in unique_loads]
plt.bar(unique_loads, load_accuracies)
plt.xlabel('Memory Load (Number of Items)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy by Memory Load')
plt.ylim(0, 100)
plt.xticks(unique_loads)
plt.grid(axis='y', alpha=0.3)
plt.savefig('tmp_scripts/accuracy_by_load.png')

# Plot response time by memory load
plt.figure(figsize=(10, 6))
load_rts = [np.mean(response_times[loads == load]) * 1000 for load in unique_loads]
plt.bar(unique_loads, load_rts)
plt.xlabel('Memory Load (Number of Items)')
plt.ylabel('Response Time (ms)')
plt.title('Response Time by Memory Load')
plt.xticks(unique_loads)
plt.grid(axis='y', alpha=0.3)
plt.savefig('tmp_scripts/rt_by_load.png')

# Plot comparison of accuracy for in-memory vs. out-of-memory probes
plt.figure(figsize=(10, 6))
in_acc = np.mean(accuracy[in_mask]) * 100
out_acc = np.mean(accuracy[out_mask]) * 100
plt.bar(['In-Memory', 'Out-of-Memory'], [in_acc, out_acc])
plt.xlabel('Probe Condition')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy by Probe Condition')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)
plt.savefig('tmp_scripts/accuracy_by_condition.png')

print("\nPlots saved to tmp_scripts directory")