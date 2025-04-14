# %% [markdown]
# AI-generated notebook: Please use with caution
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified.
# Please be cautious when interpreting the code or results.

# %% [markdown]
# # Dandiset 000673: Exploring Data for Control of Working Memory by Phase–Amplitude Coupling of Human Hippocampal Neurons
#
# This notebook provides an initial exploration of Dandiset 000673, which contains data related to the control of working memory by phase-amplitude coupling of human hippocampal neurons. The dataset includes electrophysiological recordings and behavioral data from a Sternberg task performed on human subjects.
#
# More information about the Dandiset can be found at: [https://dandiarchive.org/dandiset/000673](https://dandiarchive.org/dandiset/000673)
#
# ## Instructions
# Before running this notebook, ensure you have the following packages installed:
# ```bash
# pip install pynwb h5py remfile dandi seaborn matplotlib pandas
# ```
#
# You can install them using pip:
# ```bash
# pip install pynwb h5py remfile dandi seaborn matplotlib pandas
# ```

# %%
import pynwb
import h5py
import remfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np
from dandi.dandiapi import DandiAPIClient

# %% [markdown]
# ## 1. Introduction to the Dandiset
#
# ### 1.1. Loading Dandiset Metadata
#
# First, let's load the Dandiset metadata to get an overview of the dataset.

# %%
client = DandiAPIClient()
dandiset = client.get_dandiset("000673")
metadata = dandiset.get_metadata()
print(metadata)

# %% [markdown]
# ### 1.2. Displaying Dandiset Description
#
# Now, let's print the description of the Dandiset to understand its purpose and content.

# %%
print(metadata.description)

# %% [markdown]
# ### 1.3. Keywords
#
# Let's display the keywords associated with the Dandiset

# %%
print(metadata.keywords)

# %% [markdown]
# ## 2. Exploring the Dataset Structure
#
# ### 2.1. Listing Assets
#
# Next, let's list the assets (files) available in the Dandiset. This will help us identify the NWB files we can explore.

# %%
assets = list(dandiset.get_assets())
for asset in assets:
    print(asset.path)

# %% [markdown]
# ### 2.2. Loading an NWB File
#
# Now, let's load an NWB file from the Dandiset and explore its structure. We will use the file `sub-1/sub-1_ses-1_ecephys+image.nwb` for this demonstration.

# %%
# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# %% [markdown]
# ### 2.3. NWB File Contents
#
# Let's print the contents of the NWB file

# %%
print(nwb)

# %% [markdown]
# ### 2.4. Displaying Session Description
#
# Display the session description

# %%
print(nwb.session_description)

# %% [markdown]
# ### 2.5. Displaying Experiment Description
#
# Display the experiment description

# %%
print(nwb.experiment_description)

# %% [markdown]
# ### 2.6. Electrode Groups
#
# Let's explore the electrode groups in the NWB file

# %%
print(nwb.electrode_groups)

# %% [markdown]
# ## 3. Accessing and Visualizing Data
#
# ### 3.1. Accessing LFP Data
#
# Let's access the local field potential (LFP) data from the NWB file.

# %%
lfps = nwb.acquisition['LFPs']
print(lfps)

# %% [markdown]
# ### 3.2. Exploring LFP Data Attributes
#
# Let's look at some attributes of the LFP data, such as the starting time, rate and unit

# %%
print(f"Starting time: {lfps.starting_time}")
print(f"Rate: {lfps.rate}")
print(f"Unit: {lfps.unit}")
print(f"Data shape: {lfps.data.shape}")

# %% [markdown]
# ### 3.3. Visualizing LFP Data
#
# It is good practice to visualize the data, which can help reveal useful information about the data and assist in the discovery of errors.
# Load a small subset of the LFP data and plot it.
# Note: Loading all the LFP data would require a substantial amount of memory, so we are only loading a small subset.

# %%
# Load a subset of the LFP data
num_channels = 70  # Number of channels to plot
time_window = 5  # Time window in seconds
start_time = 0  # Starting time in seconds
start_index = int(start_time * lfps.rate)
end_index = start_index + int(time_window * lfps.rate)
lfp_data = lfps.data[start_index:end_index, :num_channels]

# Create a time vector
if lfps.timestamps is not None:
    lfp_timestamps = lfps.timestamps[start_index:end_index]
    time_vector = lfp_timestamps - lfp_timestamps[0]
else:
    time_vector = np.linspace(start_time, start_time + time_window, end_index - start_index)

# Plot the LFP data
plt.figure(figsize=(10, 6))
for i in range(num_channels):
    plt.plot(time_vector, lfp_data[:, i] + i*100 , label=f"Channel {i}") # Adding offset for each channel
plt.xlabel("Time (s)")
plt.ylabel("LFP (uV) + offset")
plt.title("Subset of LFP Data for First 70 Channels")
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.4. Accessing Event Data
#
# Accessing event data to do analysis locked to task events

# %%
events = nwb.acquisition['events']
print(events)

# %% [markdown]
# ### 3.5. Exploring Event Data Attributes
#
# Let's look at some attributes of the event data.

# %%
print(f"Timestamps unit: {events.timestamps_unit}")
print(f"Data shape: {events.data.shape}")
print(f"Timestamps shape: {events.timestamps.shape}")

# %% [markdown]
# ### 3.6. Visualizing Event Data
#
# Let's plot the event data in relation to LFP data.
# Note: Loading all the event data would require a substantial amount of memory, so we are only loading a small subset.

# %%
# Load a subset of the event data
num_events = 50  # Number of events to plot
event_data = events.data[:num_events]
event_timestamps = events.timestamps[:num_events]

# Plot the event timestamps
plt.figure(figsize=(10, 4))
plt.plot(event_timestamps, event_data, marker='o', linestyle='-', label='Events')
plt.xlabel("Time (s)")
plt.ylabel("Event Type")
plt.title("Subset of Event Data")
plt.yticks(sorted(list(set(event_data))))  # Show only unique event types
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.7. Accessing Stimulus Presentation Data
#
# Accessing data related to when the stimuli were presented.

# %%
stimulus_presentation = nwb.stimulus['StimulusPresentation']
print(stimulus_presentation)

# %% [markdown]
# ### 3.8. Exploring Stimulus Presentation Data Attributes
#
# Let's inspect some attributes about the stimulus presentations

# %%
print(f"Timestamps unit: {stimulus_presentation.timestamps_unit}")
print(f"Data shape: {stimulus_presentation.data.shape}")
print(f"Timestamps shape: {stimulus_presentation.timestamps.shape}")
print(f"Description: {stimulus_presentation.description}")

# %% [markdown]
# ### 3.9. Visualizing Stimulus Presentation Data
#
# Let's visualize the stimulus presentation data.
# Note: Loading all the stimulus presentation data would require a substantial amount of memory, so we are only loading a small subset.

# %%
# Load a subset of the stimulus presentation data
num_presentations = 50

stimulus_data = stimulus_presentation.data[:num_presentations]
stimulus_timestamps = stimulus_presentation.timestamps[:num_presentations]

# Create a basic plot
plt.figure(figsize=(12, 6))
plt.plot(stimulus_timestamps, stimulus_data, marker='o', linestyle='-', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Stimulus Index')
plt.title('Stimulus Presentation Data')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.10. Superimposing Stimulus Templates of the Same Type
#
# Let's plot a single superposition of image masks of image type 101.

# %%
# Access the StimulusTemplates images
stimulus_templates = nwb.stimulus_template["StimulusTemplates"]
image_101 = stimulus_templates.images["image_101"]
print(image_101)

# %%
# Plot the single superposition of image masks
plt.imshow(image_101[:])
plt.title('Stimulus Templates')
plt.show()