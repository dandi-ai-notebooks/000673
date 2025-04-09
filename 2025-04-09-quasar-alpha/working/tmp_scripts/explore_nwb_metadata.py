# Script to load NWB file for sub-1 session 1 and print key metadata
# This will help plan subsequent visualizations and analyses

import pynwb
import h5py
import remfile
import os

url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"

try:
    file = remfile.File(url)
    f = h5py.File(file)
    io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
    nwb = io.read()

    lines = []

    lines.append(f"session_description: {nwb.session_description}")
    lines.append(f"identifier: {nwb.identifier}")
    lines.append(f"session_start_time: {nwb.session_start_time}")
    lines.append(f"subject ID: {nwb.subject.subject_id}, species: {nwb.subject.species}, sex: {nwb.subject.sex}, age: {nwb.subject.age}")
    lines.append(f"keywords: {nwb.keywords[:]}")
    
    lines.append(f"LFP data shape: {nwb.acquisition['LFPs'].data.shape}")
    lines.append(f"LFP rate: {nwb.acquisition['LFPs'].rate}")

    lines.append(f"Events count: {nwb.acquisition['events'].data.shape[0]}")
    lines.append(f"Stimulus presentations: {nwb.stimulus['StimulusPresentation'].data.shape[0]}")
    lines.append(f"Stimulus template image keys: {list(nwb.stimulus_template['StimulusTemplates'].images.keys())[:5]} ... (only first 5 shown)")
    
    lines.append(f"Number of units: {len(nwb.units.id)}")

    electrode_table = nwb.electrodes
    lines.append(f"Electrodes table columns: {electrode_table.colnames}")
    lines.append(f"Number of electrodes: {len(electrode_table.id)}")

    output_file = "tmp_scripts/nwb_metadata_summary.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    io.close()
except Exception as e:
    with open("tmp_scripts/nwb_metadata_summary.txt", "w") as f:
        f.write(f"Error during NWB metadata exploration: {str(e)}")