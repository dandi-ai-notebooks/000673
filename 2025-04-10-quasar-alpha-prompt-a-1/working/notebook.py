# %% [markdown]
# # AI-generated exploratory notebook for DANDI dataset 000673
# 
# **Note:** This notebook was automatically generated using `dandi-notebook-gen` AI and has **not** been manually verified. Interpret the code and results with caution. Please check all steps carefully before using this for your scientific analysis.
# 
# ---

# %% [markdown]
# ## About this dataset
# 
# **Title:** *Control of working memory by phase–amplitude coupling of human hippocampal neurons*
# 
# **Citation:** Daume, Jonathan; Kaminski, Jan; Schjetnan, Andrea G. P.; et al. (2025). _Data for: Control of working memory by phase–amplitude coupling of human hippocampal neurons_. DANDI Archive. https://dandiarchive.org/dandiset/000673/draft
# 
# **Description:**
# 
# Retaining information in working memory depends on cognitive control protecting memoranda-specific activity. Here, neural recordings from human medial temporal lobe (MTL) and prefrontal cortex during working memory tasks reveal that theta-gamma phase-amplitude coupling coordinates interactions underlying this control.
# 
# For full details, consult the publication and [sample code](https://github.com/rutishauserlab/SBCAT-release-NWB).
# 
# ---
# 
# **Keywords:** cognitive neuroscience, working memory, neurophysiology, single-neurons, phase-amplitude coupling, human, intracranial
# 
# **License:** CC-BY-4.0

# %% [markdown]
# ## Setup
# 
# This notebook assumes the following Python packages are installed:
# - `dandi`
# - `pynwb`
# - `remfile`
# - `h5py`
# - `numpy`
# - `matplotlib`
# - `seaborn`
# 
# If missing, install them following the instructions at their respective websites.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
import remfile
import pynwb
from dandi.dandiapi import DandiAPIClient

sns.set_theme()

# %% [markdown]
# ## List all assets in Dandiset 000673
# 
# Here we demonstrate how to list all files (assets) in the dandiset using the DANDI API:

# %%
client = DandiAPIClient()
dandiset = client.get_dandiset("000673", "draft")
assets = list(dandiset.get_assets())
print(f"Number of assets in Dandiset: {len(assets)}")
for asset in assets[:5]:
    print(f"- {asset.path} (size: {asset.size} bytes)")

# %% [markdown]
# For illustration, we will load and analyze the NWB file below. The code can be adapted to other files as needed.
# 
# **Example file:**
# 
# `sub-1/sub-1_ses-1_ecephys+image.nwb`

# %%
nwb_url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"

file_obj = remfile.File(nwb_url)
h5file = h5py.File(file_obj)
io = pynwb.NWBHDF5IO(file=h5file, load_namespaces=True)
nwbfile = io.read()

# %% [markdown]
# ## Basic NWB metadata

# %%
print("Session:", nwbfile.session_description)
print("Start time:", nwbfile.session_start_time)
print("Subject ID:", nwbfile.subject.subject_id)
print("Subject Sex:", nwbfile.subject.sex)
print("Subject Age:", nwbfile.subject.age)
print("Institution:", nwbfile.institution)
print("Lab:", nwbfile.lab)
print("Keywords:", nwbfile.keywords[:])

# %% [markdown]
# ## Electrode table overview

# %%
etable = nwbfile.electrodes
print("Columns:", etable.colnames)
locations = etable['location'][:]
print("Unique electrode locations:", np.unique(locations))
print("Number of electrodes:", len(etable))

# %% [markdown]
# ## Stimulus Templates - number of images

# %%
stim_imgs = nwbfile.stimulus_template['StimulusTemplates'].images
print(f"Number of stimulus images: {len(stim_imgs)}")
first_key = list(stim_imgs.keys())[0]
first_img = stim_imgs[first_key].data
print("Example image shape:", first_img.shape)

# %% [markdown]
# ## Visualize some stimulus images

# %%
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
keys = list(stim_imgs.keys())
for ax, k in zip(axs.ravel(), keys[:8]):
    ax.imshow(stim_imgs[k].data)
    ax.set_title(k)
    ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Events TTL pulse counts and timing

# %%
events = nwbfile.acquisition['events']
ttl_values = events.data[:]
ttl_times = events.timestamps[:]
print("Event TTL codes present:", np.unique(ttl_values))

plt.figure(figsize=(8,3))
plt.plot(ttl_times, ttl_values, marker='.', linestyle='none')
plt.xlabel("Time (s)")
plt.ylabel("TTL code value")
plt.title("Event TTL markers over time")
plt.show()

# %% [markdown]
# ## Overview of LFP data

# %%
lfp = nwbfile.acquisition['LFPs']
print("LFP data shape:", lfp.data.shape)
print("LFP sampling rate:", lfp.rate)
print("LFP starting time:", lfp.starting_time)

# To avoid loading the entire dataset, load a snippet:
snippet = lfp.data[0:2000, :8]  # first 5s of data (2000 samples at 400Hz), first 8 electrodes

t = np.arange(snippet.shape[0]) / lfp.rate + lfp.starting_time

plt.figure(figsize=(10,6))
for i in range(snippet.shape[1]):
    plt.plot(t, snippet[:, i] + i*0.0005)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude + offset (V)")
plt.title("Example LFP traces from 8 electrodes (snippet)")
plt.show()

# %% [markdown]
# ## Units table overview and spike counts

# %%
units = nwbfile.units
print("Units columns:", units.colnames)
unit_ids = units.id[:]
print("Number of units:", len(unit_ids))

# Get spike counts for first 10 units
nspikes = [len(units['spike_times'][i]) for i in range(min(10, len(unit_ids)))]
print("Spike counts of first 10 units:", nspikes)

plt.figure(figsize=(6,4))
plt.bar(np.arange(len(nspikes)), nspikes)
plt.xlabel("Unit index")
plt.ylabel("# Spikes")
plt.title("Spike count for first 10 units")
plt.show()

# %% [markdown]
# ## Trial information

# %%
if hasattr(nwbfile, 'trials') and nwbfile.trials is not None:
    trials = nwbfile.trials
    print("Number of trials:", len(trials.id[:]))
    print("Trial columns:", trials.colnames)
else:
    print("No trial data found in this file.")

# %% [markdown]
# # Summary
# 
# This notebook provided:
# 
# - Dataset background and metadata overview
# - Instructions to list and fetch DANDI assets
# - Demonstration of how to load an NWB file using `remfile` + `h5py` + `pynwb`
# - Example exploration of subject info, electrode metadata, stimulus images, event TTLs, LFP signals, and sorted unit spike counts
# 
# To adapt this workflow:
# - Replace the example NWB file URL with other assets of interest
# - Adjust data loading according to your RAM/network, for larger/smaller subsets
# - Dig deeper into any element (e.g., units waveform features, behavioral intervals) using the [PyNWB Documentation](https://pynwb.readthedocs.io/)
# 
# ---
# 
# This concludes the exploratory notebook.