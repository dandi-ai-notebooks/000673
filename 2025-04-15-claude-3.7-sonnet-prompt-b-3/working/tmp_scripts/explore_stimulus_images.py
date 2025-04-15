# This script explores the stimulus images and their presentation during the task

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for non-image plots
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get stimulus information
stim_templates = nwb.stimulus_template['StimulusTemplates']
stim_presentation = nwb.stimulus['StimulusPresentation']

print(f"Number of stimulus images: {len(stim_templates.images)}")

# Print information about random sample of 5 images
image_names = list(stim_templates.images.keys())
sample_images = np.random.choice(image_names, size=5, replace=False)

print("\nSample of images:")
for img_name in sample_images:
    img = stim_templates.images[img_name]
    print(f"  {img_name}: shape {img.data.shape}, min={img.data.min()}, max={img.data.max()}")

# Display a few sample images
plt.figure(figsize=(15, 10))
for i, img_name in enumerate(sample_images[:4]):  # Show first 4 sample images
    img = stim_templates.images[img_name]
    plt.subplot(2, 2, i+1)
    # Don't use seaborn for image plotting
    plt.imshow(img.data)
    plt.title(f'Image: {img_name}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('tmp_scripts/sample_stimulus_images.png')

# Look at stimulus presentation data
print(f"\nStimulus presentation data shape: {stim_presentation.data.shape}")
print(f"Stimulus presentation timestamps shape: {stim_presentation.timestamps.shape}")

# Sample of presentation data
print("\nSample of stimulus presentation data (first 10 entries):")
for i in range(min(10, len(stim_presentation.data))):
    print(f"  Time {stim_presentation.timestamps[i]:.2f}s: Image index {stim_presentation.data[i]}")

# Get trial information to understand how images are used
trials = nwb.trials.to_dataframe()

# Look at a sample trial to see how images are associated
sample_trial = trials.iloc[0]
print("\nSample trial image IDs:")
print(f"  Encoding 1: {sample_trial['PicIDs_Encoding1']}")
print(f"  Encoding 2: {sample_trial['PicIDs_Encoding2']}")
print(f"  Encoding 3: {sample_trial['PicIDs_Encoding3']}")
print(f"  Probe: {sample_trial['PicIDs_Probe']}")
print(f"  Probe In/Out: {sample_trial['probe_in_out']} (1 = in list, 0 = not in list)")
print(f"  Response Accuracy: {sample_trial['response_accuracy']} (1 = correct, 0 = incorrect)")

# Plot stimulus presentation sequence for a small time window
plt.figure(figsize=(15, 5))
# Get a window of 100 stimulus presentations
window_size = 100
window_start = 0
window_end = min(window_start + window_size, len(stim_presentation.data))

# Plot stimulus indices
plt.subplot(1, 1, 1)
plt.plot(stim_presentation.timestamps[window_start:window_end], 
         stim_presentation.data[window_start:window_end], 'o-')
plt.xlabel('Time (s)')
plt.ylabel('Image Index')
plt.title('Stimulus Presentation Sequence')
plt.grid(True)

plt.tight_layout()
plt.savefig('tmp_scripts/stimulus_presentation_sequence.png')

# Calculate some statistics about image frequency
img_indices = stim_presentation.data[:]
unique_indices, counts = np.unique(img_indices, return_counts=True)

print("\nNumber of unique image indices presented:", len(unique_indices))
print(f"Most frequent image index: {unique_indices[np.argmax(counts)]} (presented {np.max(counts)} times)")
print(f"Least frequent image index: {unique_indices[np.argmin(counts)]} (presented {np.min(counts)} times)")

# Get trial statistics related to probe images
in_list_trials = trials[trials['probe_in_out'] == 1]
out_list_trials = trials[trials['probe_in_out'] == 0]

print("\nTrial statistics:")
print(f"  Total trials: {len(trials)}")
print(f"  Trials with probe in encoding list: {len(in_list_trials)} ({len(in_list_trials)/len(trials)*100:.1f}%)")
print(f"  Trials with probe not in encoding list: {len(out_list_trials)} ({len(out_list_trials)/len(trials)*100:.1f}%)")

# Calculate accuracy
accuracy_in = in_list_trials['response_accuracy'].mean()
accuracy_out = out_list_trials['response_accuracy'].mean()
total_accuracy = trials['response_accuracy'].mean()

print("\nBehavioral performance:")
print(f"  Overall accuracy: {total_accuracy:.2f}")
print(f"  Accuracy when probe in list: {accuracy_in:.2f}")
print(f"  Accuracy when probe not in list: {accuracy_out:.2f}")

# Close the file
io.close()