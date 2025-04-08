# Script to explore behavioral data from NWB file
# Shows accuracy and response times by memory load

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/000673/assets/f3a72b16-6e70-4adf-8384-28224ce212a8/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trial information
trials = nwb.intervals["trials"]
loads = trials["loads"][:]  # Memory loads (1, 2, or 3 items) 
accuracy = trials["response_accuracy"][:]  # 1=correct, 0=incorrect
probe_times = trials["timestamps_Probe"][:]
response_times = trials["timestamps_Response"][:]
rt = response_times - probe_times  # Response times (sec)

# Analyze by memory load
load_levels = [1, 2, 3]
acc_by_load = []
mean_rt_by_load = []
rt_sem_by_load = []

for load in load_levels:
    mask = loads == load
    acc = accuracy[mask].mean() * 100  # Percent correct
    acc_by_load.append(acc)
    
    # Response times for correct trials only
    rt_correct = rt[mask & (accuracy == 1)]
    mean_rt_by_load.append(rt_correct.mean())
    rt_sem_by_load.append(rt_correct.std() / np.sqrt(len(rt_correct)))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy plot
ax1.bar(load_levels, acc_by_load, color=['C0', 'C1', 'C2'])
ax1.set_title('Accuracy by Memory Load')
ax1.set_xlabel('Memory Load (items)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 100)
ax1.set_xticks(load_levels)

# Response time plot
ax2.bar(load_levels, mean_rt_by_load, color=['C0', 'C1', 'C2'],
       yerr=rt_sem_by_load, capsize=5)
ax2.set_title('Response Time by Memory Load')
ax2.set_xlabel('Memory Load (items)')
ax2.set_ylabel('Response Time (s)')
ax2.set_xticks(load_levels)

plt.tight_layout()
plt.savefig('tmp_scripts/behavior_analysis.png')
plt.close()