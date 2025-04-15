# %% [markdown]
# Exploring Dandiset 000673: Control of working memory by phase–amplitude coupling of human hippocampal neurons

# %% [markdown]
# **Important Note:** This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Use caution when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This Dandiset (000673) contains data for the study "Control of working memory by phase–amplitude coupling of human hippocampal neurons" by Daume et al. (2025). The study investigates how cognitive control regulates working memory storage through interactions of frontal control and hippocampal persistent activity coordinated by theta–gamma phase–amplitude coupling (TG-PAC). The dataset includes single-neuron recordings from the human medial temporal and frontal lobes during a working memory task.

# %% [markdown]
# ## What this notebook will cover
#
# This notebook will guide you through the process of exploring and analyzing the data in Dandiset 000673. We will cover the following steps:
#
# 1.  Connecting to the DANDI archive and loading the Dandiset metadata.
# 2.  Loading and examining the assets (NWB files) in the Dandiset.
# 3.  Loading metadata from one of the NWB files.
# 4.  Loading and visualizing electrophysiology data (LFPs) from the NWB file.
# 5.  Loading and visualizing stimulus event data from the NWB file.
# 6.  Loading and visualizing stimulus template images from the NWB file.
# 7.  Summarizing findings and suggesting future directions for analysis.

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook. Please ensure they are installed in your environment.
#
# *   pynwb
# *   h5py
# *   remfile
# *   numpy
# *   matplotlib
# *   seaborn

# %% [markdown]
# ## Loading the Dandiset using the DANDI API
#
# Here, we connect to the DANDI archive and retrieve the Dandiset 000673.

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("000673")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading metadata for an NWB file
#
# We will now load the metadata for one of the NWB files in the Dandiset. We will use the file "sub-1/sub-1\_ses-1\_ecephys+image.nwb" for this example.

# %%
import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

nwb

# %% [markdown]
# ### Examining the NWB file contents

# %%
nwb.session_description # (str) SBCAT_ID: 1

# %%
nwb.identifier # (str) sub-1_ses-1_P55CS

# %%
nwb.session_start_time # (datetime) 2018-01-01T00:00:00-08:00

# %% [markdown]
# ## Loading and visualizing LFP data
#
# Here, we load and visualize some LFP data from the NWB file.

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load LFP data
lfps = nwb.acquisition["LFPs"].data
lfps

# %%
# Visualize a subset of LFP data for the first 10 channels
num_channels = 10
num_timepoints = 1000
lfp_data = lfps[:num_timepoints, :num_channels]  # Access first 1000 timepoints and first 10 channels

plt.figure(figsize=(12, 6))
plt.plot(lfp_data)
plt.xlabel("Timepoints")
plt.ylabel("LFP (volts)")
plt.title("LFP Data for First 10 Channels")
plt.show()

# %% [markdown]
# ## Loading and visualizing stimulus event data
#
# Here, we load and visualize stimulus event data from the NWB file.

# %%
# Load stimulus event data
events_data = nwb.acquisition["events"].data
events_timestamps = nwb.acquisition["events"].timestamps

print(f"Shape of events data: {events_data.shape}")
print(f"Shape of events timestamps: {events_timestamps.shape}")

# Visualize stimulus event data
plt.figure(figsize=(12, 4))
plt.plot(events_timestamps[:100], events_data[:100], marker='o', linestyle='-')  # Plot only first 100 events
plt.xlabel("Time (seconds)")
plt.ylabel("Event Value")
plt.title("Stimulus Event Data")
plt.show()

# %% [markdown]
# ## Loading and visualizing stimulus template images
#
# Here, we load and visualize stimulus template images from the NWB file.

# %%
# Load stimulus template images
stimulus_templates = nwb.stimulus_template["StimulusTemplates"]
image_names = list(stimulus_templates.images.keys())
print(f"Available image names: {image_names}")

# Select the first image
first_image_name = image_names[0]
first_image = stimulus_templates.images[first_image_name]

print(f"Shape of the first image: {first_image.data.shape}")

# Visualize the first image
plt.figure(figsize=(4, 4))
plt.imshow(first_image.data)
plt.title(first_image_name)
plt.axis('off')  # Turn off axis labels
plt.show()

# %% [markdown]
# ## Summarizing findings and future directions
#
# In this notebook, we have demonstrated how to load and visualize data from Dandiset 000673. We have shown how to:
#
# *   Connect to the DANDI archive and load the Dandiset metadata.
# *   Load and examine the assets (NWB files) in the Dandiset.
# *   Load metadata from an NWB file.
# *   Load and visualize electrophysiology data (LFPs) from the NWB file.
# *   Load and visualize stimulus event data from the NWB file.
# *   Load and visualize stimulus template images from the NWB file.
#
# Possible future directions for analysis include:
#
# *   Performing more detailed analysis of the electrophysiology data, such as spike sorting and LFP analysis.
# *   Investigating the relationship between stimulus events and neural activity.
# *   Exploring the content of different stimulus template images and their impact on neural responses.
# *   Analyzing the data from other NWB files in the Dandiset to gain a more comprehensive understanding of the study.