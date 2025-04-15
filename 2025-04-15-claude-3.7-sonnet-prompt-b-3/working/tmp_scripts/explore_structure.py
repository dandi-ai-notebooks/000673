# This script explores the basic structure of the NWB file,
# focusing on the metadata, available data types, and basic information.

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic file information
print("===== Basic File Information =====")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experiment description: {nwb.experiment_description}")
print(f"Session ID: {nwb.session_id}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")

# Print subject information
print("\n===== Subject Information =====")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")

# Print electrode information
print("\n===== Electrode Information =====")
print(f"Number of electrodes: {len(nwb.electrodes)}")
print(f"Electrode columns: {nwb.electrodes.colnames}")

# Print a sample of electrode locations
electrode_df = nwb.electrodes.to_dataframe()
unique_locations = electrode_df['location'].unique()
print(f"\nUnique electrode locations ({len(unique_locations)}):")
for i, location in enumerate(unique_locations):
    print(f"  {i+1}. {location}")

# Print LFP data information
print("\n===== LFP Data Information =====")
lfp_data = nwb.acquisition["LFPs"]
print(f"LFP Data Shape: {lfp_data.data.shape}")
print(f"Sampling rate: {lfp_data.rate} Hz")
print(f"Starting time: {lfp_data.starting_time} {lfp_data.starting_time_unit}")
print(f"Description: {lfp_data.description}")

# Print event information
print("\n===== Event Information =====")
events = nwb.acquisition["events"]
print(f"Events Shape: {events.data.shape}")
print(f"Events Description: {events.description}")

# Print stimulus information
print("\n===== Stimulus Information =====")
stim_pres = nwb.stimulus["StimulusPresentation"]
print(f"Stimulus Presentation Shape: {stim_pres.data.shape}")
print(f"Stimulus Presentation Description: {stim_pres.description}")

# Print trial information
print("\n===== Trial Information =====")
print(f"Number of trials: {len(nwb.trials)}")
print(f"Trial columns: {nwb.trials.colnames}")
print("\nSample of trial data:")
trial_df = nwb.trials.to_dataframe().head(3)
print(trial_df)

# Print unit information
print("\n===== Unit Information =====")
print(f"Number of units: {len(nwb.units)}")
print(f"Unit columns: {nwb.units.colnames}")
print(f"Waveform unit: {nwb.units.waveform_unit}")

# Close the file
io.close()