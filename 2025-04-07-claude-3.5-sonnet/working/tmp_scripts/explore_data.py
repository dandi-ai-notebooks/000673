"""
This script explores the basic structure of the data from sub-1_ses-1, including:
1. Trial structure and behavioral performance
2. Example neural activity patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi
import seaborn as sns
from collections import Counter

# Load the data
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Set up the figure with multiple subplots
plt.figure(figsize=(15, 10))

# Plot 1: Distribution of memory loads
plt.subplot(2, 2, 1)
loads = nwb.intervals["trials"]["loads"][:]
load_counts = Counter(loads)
sns.barplot(x=list(load_counts.keys()), y=list(load_counts.values()))
plt.title('Distribution of Memory Loads')
plt.xlabel('Memory Load')
plt.ylabel('Number of Trials')

# Plot 2: Behavioral performance by memory load
plt.subplot(2, 2, 2)
accuracy_by_load = {}
for load in np.unique(loads):
    load_mask = loads == load
    accuracies = nwb.intervals["trials"]["response_accuracy"][:][load_mask]
    accuracy_by_load[load] = np.mean(accuracies) * 100

sns.barplot(x=list(accuracy_by_load.keys()), y=list(accuracy_by_load.values()))
plt.title('Accuracy by Memory Load')
plt.xlabel('Memory Load')
plt.ylabel('Accuracy (%)')

# Plot 3: Example neural activity
plt.subplot(2, 2, (3, 4))
# Get spike times for first 5 units during first 60 seconds
unit_ids = nwb.units["id"].data[:5]  # Get first 5 units
plt.title('Example Neural Activity (First 5 Units, First 60s)')
plt.xlabel('Time (s)')
plt.ylabel('Unit #')

for i, unit_id in enumerate(unit_ids):
    spike_times = nwb.units["spike_times"][i]
    spike_times = spike_times[spike_times < 60]  # First 60 seconds
    plt.plot(spike_times, np.ones_like(spike_times) * i, '|', markersize=4)

plt.tight_layout()
plt.savefig('tmp_scripts/basic_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Also save some key statistics to a text file
with open('tmp_scripts/analysis_stats.txt', 'w') as f:
    f.write(f"Total number of trials: {len(loads)}\n")
    f.write(f"Number of units: {len(nwb.units['id'].data)}\n")
    f.write("\nAccuracy by memory load:\n")
    for load, acc in accuracy_by_load.items():
        f.write(f"Load {load}: {acc:.1f}%\n")