import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# LFP data
LFPs = nwb.acquisition["LFPs"]
lfp_data = LFPs.data
lfp_rate = LFPs.rate
electrodes = LFPs.electrodes

# Trials data
trials = nwb.intervals["trials"]
trial_start_times = trials["start_time"][:]
trial_stop_times = trials["stop_time"][:]
loads = trials["loads"][:]

# Select a subset of channels and time
channel_ids = [0, 1, 2, 3, 4]
start_time = 10  # seconds
end_time = 15  # seconds

# Find the indices corresponding to the start and end times
start_index = int(start_time * lfp_rate)
end_index = int(end_time * lfp_rate)

# Extract LFP data for selected channels and time window
lfp_subset = lfp_data[start_index:end_index, channel_ids]

# Time vector for plotting
time_vector = np.linspace(start_time, end_time, lfp_subset.shape[0])

# Create the LFP plot
plt.figure(figsize=(10, 6))
for i, channel_id in enumerate(channel_ids):
    plt.plot(time_vector, lfp_subset[:, i] + i * 100, label=f"Channel {channel_id}")  # Add offset for readability

plt.xlabel("Time (s)")
plt.ylabel("LFP (uV) + offset")
plt.title("LFP Data for Selected Channels")
plt.yticks([])  # Remove y-axis ticks
plt.legend()
plt.savefig("tmp_scripts/lfp_plot.png")
plt.close()

# Create the trials plot
plt.figure(figsize=(8, 4))
plt.eventplot(trial_start_times, colors='red', label='Trial Start', linelengths=0.5)
plt.eventplot(trial_stop_times, colors='blue', label='Trial Stop', linelengths=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Trials")
plt.title("Trial Start and Stop Times")
plt.legend()
plt.savefig("tmp_scripts/trials_plot.png")
plt.close()

# loads plot
plt.figure(figsize=(8,4))
plt.plot(trial_start_times, loads, marker='o', linestyle='-', color='green')
plt.xlabel("Trial Start Time (s)")
plt.ylabel("Load")
plt.title("Trial Loads vs Start Time")
plt.savefig("tmp_scripts/loads_plot.png")
plt.close()