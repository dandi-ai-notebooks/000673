"""
This script analyzes the stimulus images used in the Sternberg working memory task
and examines how they were presented during the experiment.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

# Ensure the output directory exists
os.makedirs("tmp_scripts", exist_ok=True)

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/9fdbe18f-158f-47c5-ba67-4c56118d6cf5/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Access the stimulus presentation data
StimulusPresentation = nwb.stimulus["StimulusPresentation"]
trials = nwb.intervals["trials"]

# Print basic stimulus information
print("Stimulus Information:")
print("-----------------------")
print(f"Total number of stimulus presentations: {len(StimulusPresentation.data)}")

# Get information about the image stimuli
indexed_images = StimulusPresentation.indexed_images
image_keys = list(indexed_images.images.keys())
print(f"Number of unique images: {len(image_keys)}")
print(f"Sample image keys: {image_keys[:5]}")

# Check the image IDs used in trials
pic_ids_enc1 = trials["PicIDs_Encoding1"][:]
pic_ids_enc2 = trials["PicIDs_Encoding2"][:]
pic_ids_enc3 = trials["PicIDs_Encoding3"][:]
pic_ids_probe = trials["PicIDs_Probe"][:]

print("\nTrial Image IDs:")
print(f"Encoding1 IDs range: {np.min(pic_ids_enc1)} to {np.max(pic_ids_enc1)}")
print(f"Encoding2 IDs range: {np.min(pic_ids_enc2)} to {np.max(pic_ids_enc2)}")
print(f"Encoding3 IDs range: {np.min(pic_ids_enc3)} to {np.max(pic_ids_enc3)}")
print(f"Probe IDs range: {np.min(pic_ids_probe)} to {np.max(pic_ids_probe)}")

# Display a few of the available images
sample_keys = image_keys[:5]  # Display the first 5 images
print(f"\nDisplaying sample images with keys: {sample_keys}")

# Plot sample images
plt.figure(figsize=(15, 10))
for i, key in enumerate(sample_keys):
    image_data = indexed_images.images[key].data[:]
    
    plt.subplot(2, 3, i+1)
    plt.imshow(image_data, cmap='gray')
    plt.title(f"Image Key: {key}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('tmp_scripts/sample_images.png')

# Analyze how the stimulus data and trial data are connected
stimulus_data = StimulusPresentation.data[:]
stimulus_timestamps = StimulusPresentation.timestamps[:]

# Look at the data values in StimulusPresentation
print(f"\nStimulusPresentation data range: {np.min(stimulus_data)} to {np.max(stimulus_data)}")
print(f"First few data values: {stimulus_data[:10]}")

# Check if image indices match between StimulusPresentation and the image keys
indices_in_stimulus = set(stimulus_data)
indices_in_images = set(range(len(image_keys)))

print(f"\nDo stimulus indices match image indices? {indices_in_stimulus == indices_in_images}")
print(f"Number of unique indices in stimulus data: {len(indices_in_stimulus)}")

# Analyze the mapping between stimulus indices and trial picture IDs
# We need to understand how to match the trial pic IDs to the actual images

# Look at the distribution of stimulus presentations
all_pic_ids = np.concatenate([pic_ids_enc1, pic_ids_enc2, pic_ids_enc3, pic_ids_probe])
unique_pic_ids, pic_counts = np.unique(all_pic_ids, return_counts=True)

print("\nImage Presentation Statistics:")
print(f"Unique image IDs used in trials: {len(unique_pic_ids)}")
print(f"Total image presentations in trials: {len(all_pic_ids)}")
print(f"Average presentations per image: {len(all_pic_ids)/len(unique_pic_ids):.1f}")

# Analyze trial timing information
enc1_times = trials["timestamps_Encoding1"][:]
enc1_end_times = trials["timestamps_Encoding1_end"][:]
enc2_times = trials["timestamps_Encoding2"][:]
enc2_end_times = trials["timestamps_Encoding2_end"][:]
enc3_times = trials["timestamps_Encoding3"][:]
enc3_end_times = trials["timestamps_Encoding3_end"][:]
maint_times = trials["timestamps_Maintenance"][:]
probe_times = trials["timestamps_Probe"][:]

# Calculate average durations
enc1_duration = np.mean(enc1_end_times - enc1_times)
load3_mask = trials["loads"][:] == 3
if np.any(load3_mask):  # Only if load 3 trials exist
    enc2_duration = np.mean(enc2_end_times[load3_mask] - enc2_times[load3_mask])
    enc3_duration = np.mean(enc3_end_times[load3_mask] - enc3_times[load3_mask])
    encoding_intervals = np.mean(enc2_times[load3_mask] - enc1_end_times[load3_mask])
else:
    enc2_duration = np.nan
    enc3_duration = np.nan
    encoding_intervals = np.nan

maint_duration = np.mean(probe_times - maint_times)

# Print timing information
print("\nTrial Timing Information:")
print(f"Average encoding 1 duration: {enc1_duration*1000:.1f} ms")
print(f"Average encoding 2 duration: {enc2_duration*1000:.1f} ms")
print(f"Average encoding 3 duration: {enc3_duration*1000:.1f} ms")
print(f"Average interval between encodings: {encoding_intervals*1000:.1f} ms")
print(f"Average maintenance period duration: {maint_duration*1000:.1f} ms")

# Create a plot showing the timing of trial events
plt.figure(figsize=(12, 6))

# Plot for load 3 trial
trial_idx = np.where(load3_mask)[0][0]
times = [enc1_times[trial_idx], enc1_end_times[trial_idx], 
         enc2_times[trial_idx], enc2_end_times[trial_idx],
         enc3_times[trial_idx], enc3_end_times[trial_idx],
         maint_times[trial_idx], probe_times[trial_idx]]
events = ["Enc1 Start", "Enc1 End", "Enc2 Start", "Enc2 End", 
          "Enc3 Start", "Enc3 End", "Maintenance", "Probe"]
times = [t - times[0] for t in times]  # Convert to relative times

plt.figure(figsize=(12, 4))
plt.plot(times, np.ones(len(times)), 'o', markersize=10)
for i, (t, e) in enumerate(zip(times, events)):
    plt.text(t, 1.05, e, rotation=45, ha='right')
for i in range(len(times)-1):
    plt.axvline(times[i], color='gray', linestyle='--', alpha=0.3)
plt.title("Timeline of Trial Events (Load 3)")
plt.xlabel("Time from Trial Start (seconds)")
plt.yticks([])
plt.ylim(0.8, 1.2)
plt.tight_layout()
plt.savefig('tmp_scripts/trial_timeline.png')

print("\nPlots saved to tmp_scripts directory")