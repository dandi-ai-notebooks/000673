"""
This script explores the visual stimuli in the NWB file, including stimulus images
and their presentation during the Sternberg task.
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

# Get the stimulus templates
print("Extracting stimulus information...")
stimuli = nwb.stimulus_template["StimulusTemplates"]
image_keys = list(stimuli.images.keys())
print(f"Number of stimulus images: {len(image_keys)}")
print(f"Image IDs: {image_keys[:5]} ... (showing first 5 of {len(image_keys)})")

# Get the stimulus presentation information
stim_presentation = nwb.stimulus["StimulusPresentation"]
print(f"Number of stimulus presentations: {len(stim_presentation.data)}")

# Get a few example images to display
sample_image_keys = image_keys[:4]  # Just get the first 4 images
sample_images = []

for key in sample_image_keys:
    img = stimuli.images[key].data[:]
    sample_images.append(img)
    
# Plot the sample images
plt.figure(figsize=(12, 6))
for i, (key, img) in enumerate(zip(sample_image_keys, sample_images)):
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(f"Image ID: {key}")
    plt.axis('off')
    
plt.tight_layout()
plt.savefig('explore/sample_stimuli.png', dpi=300, bbox_inches='tight')

# Get trial information
trials = nwb.intervals['trials']
trials_df = trials.to_dataframe()

# Look at stimulus use in trials
print("\nAnalyzing stimulus use in trials...")
stim_counts = {}

# Count usage of each image as encoding item 1
enc1_counts = trials_df['PicIDs_Encoding1'].value_counts().to_dict()
for pic_id, count in enc1_counts.items():
    if pic_id not in stim_counts:
        stim_counts[pic_id] = 0
    stim_counts[pic_id] += count

# Count usage of each image as encoding item 2
enc2_counts = trials_df['PicIDs_Encoding2'].value_counts().to_dict()
for pic_id, count in enc2_counts.items():
    if pic_id not in stim_counts:
        stim_counts[pic_id] = 0
    stim_counts[pic_id] += count

# Count usage of each image as encoding item 3
enc3_counts = trials_df['PicIDs_Encoding3'].value_counts().to_dict()
for pic_id, count in enc3_counts.items():
    if pic_id not in stim_counts:
        stim_counts[pic_id] = 0
    stim_counts[pic_id] += count

# Count usage of each image as probe
probe_counts = trials_df['PicIDs_Probe'].value_counts().to_dict()
for pic_id, count in probe_counts.items():
    if pic_id not in stim_counts:
        stim_counts[pic_id] = 0
    stim_counts[pic_id] += count

# Sort by total count
sorted_stim_counts = sorted(stim_counts.items(), key=lambda x: x[1], reverse=True)
top_stimuli = sorted_stim_counts[:20]  # Top 20 most frequently used images

print(f"Top 5 most frequently used images: {top_stimuli[:5]}")

# Plot top stimuli usage
plt.figure(figsize=(12, 6))
top_ids = [str(s[0]) for s in top_stimuli]
top_counts = [s[1] for s in top_stimuli]

plt.bar(top_ids, top_counts)
plt.xlabel('Stimulus ID')
plt.ylabel('Usage Count')
plt.title('Top 20 Most Frequently Used Stimuli')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('explore/top_stimuli_usage.png', dpi=300, bbox_inches='tight')

# Analyze stimulus presentation during trials
print("\nAnalyzing stimulus presentation timing...")
stim_times = stim_presentation.timestamps[:]
stim_ids = stim_presentation.data[:]

# Convert stimulus IDs to actual image IDs if needed
# This would depend on how they're indexed

# Plot the timing of stimulus presentations for a subset of trials
plt.figure(figsize=(14, 6))
plt.plot(stim_times[:100], stim_ids[:100], 'o-')
plt.xlabel('Time (seconds)')
plt.ylabel('Stimulus ID')
plt.title('Stimulus Presentation Timing (First 100 Presentations)')
plt.savefig('explore/stim_presentation_timing.png', dpi=300, bbox_inches='tight')

print("Analysis complete. Plots saved to 'explore' directory")