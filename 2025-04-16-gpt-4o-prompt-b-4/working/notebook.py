# %% [markdown]
# # Exploring Dandiset 000673: Control of Working Memory by Phase-Amplitude Coupling
#
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.
#
# ## Overview of the Dandiset
#
# Dandiset ID: 000673
#
# - **Name**: Data for: Control of working memory by phase–amplitude coupling of human hippocampal neurons
# - **Description**: Retaining information in working memory relies on cognitive control to protect memoranda-specific persistent activity from interference. We show that interactions of frontal control and hippocampal persistent activity are coordinated by theta–gamma phase–amplitude coupling (TG-PAC).
# - **Access**: Open Access
# - **License**: CC-BY-4.0
# - **Link**: [Neurosift Link](https://neurosift.app/dandiset/001176/000673)
#
# ## Contents
#
# This notebook explores the Local Field Potential (LFP) data from one of the NWB files within this Dandiset.

# %% [markdown]
# ## Required Packages
#
# Ensure that the following packages are installed in your environment:
# - `pynwb`
# - `h5py`
# - `matplotlib`
# - `remfile`
# - `numpy`

# %%
from dandi.dandiapi import DandiAPIClient
import remfile
import h5py
import pynwb
import matplotlib.pyplot as plt

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001174")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading the Dandiset and Selecting an NWB File

# %%
# Load the remote NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

nwb # Display some metadata

# %% [markdown]
# ## Visualizing LFP Data

# %%
# Access the LFP data
lfp_data = nwb.acquisition['LFPs'].data[:10, :10]  # Access a subset of data for visualization

# %% 
# Plotting the LFP data
plt.figure(figsize=(10, 6))
for i in range(lfp_data.shape[1]):
    plt.plot(lfp_data[:, i], label=f'Channel {i}')

plt.title('LFP Signals from Select Electrodes')
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# The LFP signals from a subset of electrodes reveal the variability across channels. Future directions include exploring more electrode data, analyzing stimulus-response relationships, and investigating phase-amplitude coupling metrics. This notebook serves as a starting point for loading and examining NWB data using DANDI.