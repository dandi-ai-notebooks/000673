import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Plot Stimulus Templates
stimulus_templates = nwb.stimulus_template["StimulusTemplates"].images
num_stimuli = len(stimulus_templates.keys())

for i, stimulus_key in enumerate(list(stimulus_templates.keys())[:5]):  # Plot the first 5 stimuli
    stimulus = stimulus_templates[stimulus_key]
    plt.figure(figsize=(5, 5))
    plt.imshow(stimulus[:])  # Display the image
    plt.title(f"Stimulus Template - {stimulus_key}")
    plt.axis("off")  # Turn off axis labels
    plt.savefig(f"tmp_scripts/stimulus_{stimulus_key}.png")  # Save the plot
    plt.close()