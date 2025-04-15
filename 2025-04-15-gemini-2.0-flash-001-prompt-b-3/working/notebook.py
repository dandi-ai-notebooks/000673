# %% [markdown]
# # Exploring Dandiset 000673: Control of working memory by phase–amplitude coupling of human hippocampal neurons
#
# **Important Note:** This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Use caution when interpreting the code or results.
#
# ## Overview of the Dandiset
#
# This Dandiset (000673) contains data related to the study of working memory and phase-amplitude coupling in human hippocampal neurons. The data includes electrophysiological recordings and behavioral data from a Sternberg task.
#
# The description of the dandiset is:
# >Retaining information in working memory is a demanding process that relies on cognitive control to protect memoranda-specific persistent activity from interference. However, how cognitive control regulates working memory storage is unclear. Here we show that interactions of frontal control and hippocampal persistent activity are coordinated by theta–gamma phase–amplitude coupling (TG-PAC). We recorded single neurons in the human medial temporal and frontal lobe while patients maintained multiple items in their working memory. In the hippocampus, TG-PAC was indicative of working memory load and quality. We identified cells that selectively spiked during nonlinear interactions of theta phase and gamma amplitude. The spike timing of these PAC neurons was coordinated with frontal theta activity when cognitive control demand was high. By introducing noise correlations with persistently active neurons in the hippocampus, PAC neurons shaped the geometry of the population code. This led to higher-fidelity representations of working memory content that were associated with improved behaviour. Our results support a multicomponent architecture of working memory, with frontal control managing maintenance of working memory content in storage-related areas. Within this framework, hippocampal TG-PAC integrates cognitive control and working memory storage across brain areas, thereby suggesting a potential mechanism for top-down control over sensory-driven processes.
#
# ## Notebook Overview
#
# This notebook will guide you through the process of exploring the data in Dandiset 000673. We will cover the following:
#
# 1.  Loading the Dandiset using the DANDI API
# 2.  Loading metadata for an NWB file in the Dandiset
# 3.  Loading and visualizing LFP data from the NWB file
# 4.  Loading and visualizing stimulus template images from the NWB file
#
# ## Required Packages
#
# The following packages are required to run this notebook:
#
# *   pynwb
# *   h5py
# *   remfile
# *   matplotlib
# *   numpy
# *   seaborn
#
# Assume that these are already installed on the user's system.
#
# %%
# Connect to DANDI archive
from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("000673")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading Metadata for an NWB File
#
# We will now load the metadata for one of the NWB files in the Dandiset. We will use the file `sub-1/sub-1_ses-1_ecephys+image.nwb` as an example.

# %%
# Load the NWB file
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

nwb

# %% [markdown]
# The following are some examples of how to access some of the metadata from the NWB file:

# %%
nwb.session_description  # (str) SBCAT_ID: 1
nwb.identifier  # (str) sub-1_ses-1_P55CS
nwb.session_start_time

# %% [markdown]
# ## Loading and Visualizing LFP Data
# This section demonstrates how to load and visualize LFP data from the NWB file.

# %%
# Plot LFP data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme() # Restore seaborn theme for plot aesthetics

lfp_data = nwb.acquisition["LFPs"].data
lfp_rate = nwb.acquisition["LFPs"].rate
num_channels = lfp_data.shape[1]
time = np.arange(0, 10, 1 / lfp_rate)  # Time in seconds for first 10 seconds
num_timepoints = len(time)

for i in range(min(5, num_channels)):  # Plot the first 5 channels
    plt.figure(figsize=(10, 2))
    plt.plot(time, lfp_data[0:num_timepoints, i])
    plt.xlabel("Time (s)")
    plt.ylabel("LFP (volts)")
    plt.title(f"LFP Data - Channel {i}")
    plt.show()
    # plt.savefig(f"tmp_scripts/lfp_channel_{i}.png")  # No longer saving figures: Plots are displayed

# %% [markdown]
# The plot above show the LFP data for the first 5 channels.

# %% [markdown]
# ## Loading and Visualizing Stimulus Template Images
# This section demonstrates how to load and visualize stimulus template images from the NWB file.

# %%
# Plot Stimulus Templates
import matplotlib.pyplot as plt
import numpy as np

stimulus_templates = nwb.stimulus_template["StimulusTemplates"].images
num_stimuli = len(stimulus_templates.keys())

for i, stimulus_key in enumerate(list(stimulus_templates.keys())[:5]):  # Plot the first 5 stimuli
    stimulus = stimulus_templates[stimulus_key]
    plt.figure(figsize=(5, 5))
    plt.imshow(stimulus[:])  # Display the image
    plt.title(f"Stimulus Template - {stimulus_key}")
    plt.axis("off")  # Turn off axis labels
    plt.show()
    # plt.savefig(f"tmp_scripts/stimulus_{stimulus_key}.png")  # Save the plot

# %% [markdown]
# The plots above show the first five of the stimulus template images.

# %% [markdown]
#
# ## Summary and Future Directions
# In this notebook, we have shown how to load and visualize data from Dandiset 000673, including LFP signals and stimulus template images. This is just a starting point. Possible future directions for analysis include:
#
# *   Analyzing the relationship between LFP signals and stimulus presentation.
# *   Performing spike sorting to identify individual neurons.
# *   Investigating phase-amplitude coupling between different brain regions.
# *   Comparing neural activity during different phases of the Sternberg task.