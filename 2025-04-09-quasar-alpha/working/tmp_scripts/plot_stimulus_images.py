# Extracts and saves example stimulus template images from the NWB file

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import os

url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"

try:
    file = remfile.File(url)
    f = h5py.File(file)
    io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
    nwb = io.read()

    images = nwb.stimulus_template['StimulusTemplates'].images
    keys = sorted(images.keys())[:3]  # just a subset
    
    os.makedirs("tmp_scripts", exist_ok=True)
    for key in keys:
        img_data = images[key].data[:]
        plt.imshow(img_data)
        plt.axis('off')
        plt.title(f"Stimulus {key}")
        plt.savefig(f"tmp_scripts/{key}.png")
        plt.close()
    
    io.close()
except Exception as e:
    with open("tmp_scripts/stimulus_images_error.txt", "w") as f:
        f.write(str(e))