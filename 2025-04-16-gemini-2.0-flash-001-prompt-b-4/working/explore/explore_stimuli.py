# explore/explore_stimuli.py
# This script explores the stimulus templates in sub-1/sub-1_ses-1_ecephys+image.nwb.
# It plots a subset of the images in a grid.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the stimulus templates
stimulus_templates = nwb.stimulus_template["StimulusTemplates"].images

# Select a subset of images
num_images = min(10, len(stimulus_templates))
image_names = list(stimulus_templates.keys())[:num_images]

# Plot these images in a grid
plt.figure(figsize=(12, 6))
for i, image_name in enumerate(image_names):
    plt.subplot(2, 5, i + 1)
    image_data = stimulus_templates[image_name].data[:]
    image = Image.fromarray(image_data)
    plt.imshow(image)
    plt.title(image_name)
    plt.axis("off")

plt.tight_layout()
plt.savefig("explore/stimulus_templates.png")
plt.close()