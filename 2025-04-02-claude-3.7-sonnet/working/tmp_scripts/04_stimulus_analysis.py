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
image_ids = list(indexed_images.images.keys())
print(f"Number of unique images: {len(image_ids)}")

# Choose a few images to visualize
sample_image_ids = image_ids[:5]  # Display the first 5 images
print(f"Displaying sample images with IDs: {sample_image_ids}")

# Plot sample images
plt.figure(figsize=(15, 10))
for i, image_id in enumerate(sample_image_ids):
    image_data = indexed_images.images[image_id].data[:]
    
    plt.subplot(2, 3, i+1)
    plt.imshow(image_data, cmap='gray')
    plt.title(f"Image ID: {image_id}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('tmp_scripts/sample_images.png')

# Analyze image presentation during trials
pic_ids_enc1 = trials["PicIDs_Encoding1"][:]
pic_ids_enc2 = trials["PicIDs_Encoding2"][:]
pic_ids_enc3 = trials["PicIDs_Encoding3"][:]
pic_ids_probe = trials["PicIDs_Probe"][:]

# Count occurrences of each image
all_pic_ids = np.concatenate([pic_ids_enc1, pic_ids_enc2, pic_ids_enc3, pic_ids_probe])
unique_pic_ids, pic_counts = np.unique(all_pic_ids, return_counts=True)

# How many images appear in each position
print("\nImage Presentation Statistics:")
print(f"Unique images used in trials: {len(unique_pic_ids)}")
print(f"Total image presentations in trials: {len(all_pic_ids)}")
print(f"Average presentations per image: {len(all_pic_ids)/len(unique_pic_ids):.1f}")

# Analyze how images are used in the task
in_memory = trials["probe_in_out"][:]
accuracy = trials["response_accuracy"][:]

# For load 1 trials (only 1 encoding image)
load1_mask = trials["loads"][:] == 1
load3_mask = trials["loads"][:] == 3

# Create plots of trial structure
plt.figure(figsize=(10, 6))
# Choose a random load 3 trial to visualize
trial_idx = np.where(load3_mask)[0][0]

# Get image IDs for this trial
enc1_id = pic_ids_enc1[trial_idx]
enc2_id = pic_ids_enc2[trial_idx]
enc3_id = pic_ids_enc3[trial_idx]
probe_id = pic_ids_probe[trial_idx]

# Display images for this trial
plt.figure(figsize=(15, 5))
# First encoding image
plt.subplot(1, 4, 1)
plt.imshow(indexed_images.images[str(enc1_id)].data[:], cmap='gray')
plt.title(f"Encoding 1\nID: {enc1_id}")
plt.axis('off')

# Second encoding image
plt.subplot(1, 4, 2)
plt.imshow(indexed_images.images[str(enc2_id)].data[:], cmap='gray')
plt.title(f"Encoding 2\nID: {enc2_id}")
plt.axis('off')

# Third encoding image
plt.subplot(1, 4, 3)
plt.imshow(indexed_images.images[str(enc3_id)].data[:], cmap='gray')
plt.title(f"Encoding 3\nID: {enc3_id}")
plt.axis('off')

# Probe image
plt.subplot(1, 4, 4)
plt.imshow(indexed_images.images[str(probe_id)].data[:], cmap='gray')
in_mem = "In Memory" if in_memory[trial_idx] else "Not In Memory"
plt.title(f"Probe\nID: {probe_id}\n{in_mem}")
plt.axis('off')

plt.suptitle(f"Sample Trial (Load 3) - Trial {trial_idx+1}")
plt.savefig('tmp_scripts/sample_trial_load3.png')

# Do the same for a load 1 trial
plt.figure(figsize=(15, 5))
trial_idx = np.where(load1_mask)[0][0]

# Get image IDs for this trial
enc1_id = pic_ids_enc1[trial_idx]
probe_id = pic_ids_probe[trial_idx]

# First encoding image
plt.subplot(1, 2, 1)
plt.imshow(indexed_images.images[str(enc1_id)].data[:], cmap='gray')
plt.title(f"Encoding 1\nID: {enc1_id}")
plt.axis('off')

# Probe image
plt.subplot(1, 2, 2)
plt.imshow(indexed_images.images[str(probe_id)].data[:], cmap='gray')
in_mem = "In Memory" if in_memory[trial_idx] else "Not In Memory"
plt.title(f"Probe\nID: {probe_id}\n{in_mem}")
plt.axis('off')

plt.suptitle(f"Sample Trial (Load 1) - Trial {trial_idx+1}")
plt.savefig('tmp_scripts/sample_trial_load1.png')

# Analyze temporal structure of trials
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

print("\nPlots saved to tmp_scripts directory")