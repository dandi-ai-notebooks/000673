"""
This script explores the structure of the NWB file from Dandiset 000673,
focusing on the general organization and metadata available.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Print basic information about the dataset
print("\n=== BASIC INFORMATION ===")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Experiment Description: {nwb.experiment_description}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")

# Print subject information
print("\n=== SUBJECT INFORMATION ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Species: {nwb.subject.species}")

# Print information about the electrodes
print("\n=== ELECTRODE INFORMATION ===")
print(f"Number of electrodes: {len(nwb.electrodes)}")
print(f"Electrode columns: {nwb.electrodes.colnames}")
electrodes_df = nwb.electrodes.to_dataframe()
# Count electrodes by location
location_counts = electrodes_df['location'].value_counts()
print("\nElectrode locations:")
for location, count in location_counts.items():
    print(f"  {location}: {count}")

# Print information about the units (neurons)
print("\n=== UNITS INFORMATION ===")
print(f"Number of units: {len(nwb.units)}")
print(f"Units columns: {nwb.units.colnames}")
if len(nwb.units) > 0:
    units_df = nwb.units.to_dataframe()
    if 'electrodes' in units_df.columns:
        # Get unit count per electrode
        units_per_electrode = units_df['electrodes'].value_counts()
        print(f"\nTop 10 electrodes by unit count:")
        print(units_per_electrode.head(10))

# Print information about the trials
print("\n=== TRIALS INFORMATION ===")
print(f"Number of trials: {len(nwb.trials)}")
print(f"Trial columns: {nwb.trials.colnames}")

# Print information about the stimulus
print("\n=== STIMULUS INFORMATION ===")
if hasattr(nwb, 'stimulus') and len(nwb.stimulus) > 0:
    print(f"Stimulus modules: {list(nwb.stimulus.keys())}")
    for key in nwb.stimulus.keys():
        print(f"  {key} description: {nwb.stimulus[key].description}")

if hasattr(nwb, 'stimulus_template') and len(nwb.stimulus_template) > 0:
    print(f"\nStimulus template modules: {list(nwb.stimulus_template.keys())}")
    for key in nwb.stimulus_template.keys():
        print(f"  {key} description: {nwb.stimulus_template[key].description}")
        if hasattr(nwb.stimulus_template[key], 'images'):
            print(f"  {key} number of images: {len(nwb.stimulus_template[key].images)}")
            image_sample = list(nwb.stimulus_template[key].images.keys())[:5]
            print(f"  {key} sample images: {image_sample}...")

# Print information about the LFP data
print("\n=== LFP DATA INFORMATION ===")
if "LFPs" in nwb.acquisition:
    lfp = nwb.acquisition["LFPs"]
    print(f"LFP data shape: {lfp.data.shape}")
    print(f"LFP sampling rate: {lfp.rate} Hz")
    print(f"LFP description: {lfp.description}")
    print(f"LFP unit: {lfp.unit}")

# Print information about the events
print("\n=== EVENTS INFORMATION ===")
if "events" in nwb.acquisition:
    events = nwb.acquisition["events"]
    print(f"Events data shape: {events.data.shape}")
    print(f"Events description: {events.description}")
    # Count event types
    if len(events.data) > 0:
        event_types = np.unique(events.data)
        print(f"Event types: {event_types}")
        event_counts = {int(event_type): np.sum(events.data == event_type) for event_type in event_types}
        print(f"Event counts: {event_counts}")

# Create a figure showing the electrode locations
plt.figure(figsize=(10, 8))
unique_locations = electrodes_df['location'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_locations)))
color_map = {loc: colors[i] for i, loc in enumerate(unique_locations)}

# Use x and y coordinates for display
plt.scatter(electrodes_df['x'], electrodes_df['y'], 
           c=[color_map[loc] for loc in electrodes_df['location']], 
           alpha=0.7, s=50)

plt.title('Electrode Locations (X-Y view)')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.grid(True, alpha=0.3)

# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                     markerfacecolor=color_map[loc], markersize=10) 
          for loc in unique_locations]
plt.legend(handles, unique_locations, title='Location', loc='upper right')

plt.savefig('tmp_scripts/electrode_locations_xy.png', dpi=300, bbox_inches='tight')

# Create a figure showing the electrode locations in 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each location with a different color
for i, location in enumerate(unique_locations):
    subset = electrodes_df[electrodes_df['location'] == location]
    ax.scatter(subset['x'], subset['y'], subset['z'], 
              c=[color_map[location]], label=location, alpha=0.7, s=50)

ax.set_title('Electrode Locations (3D view)')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Z coordinate')
ax.legend(title='Location')

plt.savefig('tmp_scripts/electrode_locations_3d.png', dpi=300, bbox_inches='tight')

# Close the file
io.close()
f.close()
file.close()