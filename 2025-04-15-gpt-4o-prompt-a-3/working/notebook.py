# %% [markdown]
# # Exploring Dandiset 000673: Control of Working Memory
#
# This notebook is AI-generated using dandi-notebook-gen and has not been fully verified. Users should be cautious when interpreting the code or results.
#
# ## Overview of Dandiset 000673
#
# This Dandiset is devoted to understanding the control of working memory by phase–amplitude coupling of human hippocampal neurons. It involves datasets recorded from the medial temporal and frontal lobes.
#
# **Keywords**: cognitive neuroscience, working memory, neurophysiology, single-neurons, phase-amplitude coupling
#
# ## Contents of the Notebook
#
# 1. Load the dataset using the DANDI API.
# 2. Explore the metadata of the dataset.
# 3. Visualize selected data from the dataset.
# 4. Discuss future directions for analysis.

# %% [markdown]
# ## Required Packages
# The following packages are required to run the notebook. Ensure they are installed on your system: `pynwb`, `h5py`, `remfile`, `pandas`, `matplotlib`, `numpy`

# %%
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Load Dandiset 000673 Using DANDI API

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("000673")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Load and Explore Metadata from NWB File
# We will load the metadata using PyNWB.

# %%
# Load the NWB file using provided URL
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Explore metadata
meta_details = {
    "Session Description": nwb.session_description,
    "Identifier": nwb.identifier,
    "Session Start Time": nwb.session_start_time,
    "Experimenter": nwb.experimenter,
}
print(meta_details)

# %% [markdown]
# ## Visualize Data from NWB File
# Let's load and visualize the Local Field Potential (LFP) data and events.

# %%
# Plot LFP data - showing the first 10 channels and first 1000 data points
plt.figure(figsize=(15, 4))
lfp_data = nwb.acquisition["LFPs"].data[:1000, :10]  # Load subset of data for visualization
plt.plot(lfp_data)
plt.title("LFP Data (subset)")
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (V)')

# %% [markdown]
# ## Advanced Visualization: Stimulus Presentation
# We will also visualize the timestamps and sequence of stimulus presentations from the dataset.

# %%
# Visualize events
events_data = nwb.acquisition["events"].data[:20]  # Load subset of events data
plt.figure(figsize=(10, 2))
plt.plot(events_data, 'o-')
plt.title("Events Data (subset)")
plt.xlabel('Event Index')
plt.ylabel('Event Type')

# %% [markdown]
# ## Conclusion
# In this notebook, we've loaded and visualized LFP data and event markers. Future analysis could involve more advanced data processing and integration of multiple datasets found in this Dandiset.
#
# Some potential outcomes may revolve around understanding phase-amplitude coupling and its role in brain functions related to working memory.