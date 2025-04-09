# Plot a snippet of LFP data from sub-1 session 1 NWB for visualization
# Saves a PNG with multiple channel LFP traces over a short window

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import os

url = "https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/"

try:
    file = remfile.File(url)
    f = h5py.File(file)
    io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
    nwb = io.read()
    
    lfp = nwb.acquisition['LFPs']
    rate = lfp.rate  # Hz
    
    data = lfp.data
    
    num_samples = int(rate * 2)  # 2 seconds
    channel_idxs = np.arange(min(10, data.shape[1]))  # up to 10 channels
    
    # load a small chunk: shape (num_samples, num_channels)
    snippet = data[0:num_samples, channel_idxs]
    
    tvec = np.arange(num_samples) / rate  # seconds
    
    plt.figure(figsize=(12, 6))
    for i, ch in enumerate(channel_idxs):
        plt.plot(tvec, snippet[:, i] * 1e6 + i * 200, label=f'Ch {ch}')  # scale to uV, offset
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude + offset (uV)')
    plt.title('LFP examples from multiple channels (first 2 s)')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    
    os.makedirs("tmp_scripts", exist_ok=True)
    plt.savefig("tmp_scripts/lfp_example.png")
    plt.close()
    
    io.close()
except Exception as e:
    with open("tmp_scripts/lfp_example_error.txt", "w") as f:
        f.write(str(e))