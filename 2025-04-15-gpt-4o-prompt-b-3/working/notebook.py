# %% [markdown]
# # Exploring Dandiset 000673: Control of Working Memory by Phase–Amplitude Coupling of Human Hippocampal Neurons
#
# **Note**: This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview
# This Dandiset explores the role of phase–amplitude coupling in the regulation of working memory. The dataset includes recordings from the medial temporal lobe and frontal lobe during a working memory task.

# %% [markdown]
# The research involves:
# - Recording single neuron activities in human subjects performing cognitive tasks.
# - Exploring interactions using theta–gamma phase–amplitude coupling.
# - Utilizing LFP and event data for analysis.

# %% [markdown]
# ## Required Packages
# Ensure the following packages are installed: `pynwb`, `h5py`, `numpy`, `matplotlib`, `remfile`.

# %% [markdown]
# ## Load Dandiset
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
# ## Load and Explore LFP Data
import numpy as np
import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile

# URL to NWB file from sub-1 session 1
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"

# Use remfile to handle remote file access
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, "r")
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb_data = io.read()

# Access LFP data
lfp_series = nwb_data.acquisition["LFPs"]

# Extract data for the first 10 channels and first 1000 timepoints
selected_data = lfp_series.data[:1000, :10]

# Create a time array for plotting
time = np.arange(selected_data.shape[0]) / lfp_series.rate

# Plot the selected LFP data
plt.figure(figsize=(15, 6))
for i, channel_data in enumerate(selected_data.T):
    plt.plot(time, channel_data + i*0.1, label=f"Channel {i+1}")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("LFP Data: First 10 Channels")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary and Future Directions
# This analysis provides a starting point for exploring the Dandiset. Future work could include deeper analyses of phase–amplitude coupling across different cognitive tasks and comparisons between individuals.

io.close()
h5_file.close()