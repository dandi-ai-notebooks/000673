# Plot histogram of behavioral event timestamps from the NWB file
# Save figure to file, no plt.show()

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

    timestamps = nwb.acquisition["events"].timestamps[:]
    
    plt.hist(timestamps, bins=50, color='skyblue', edgecolor='k')
    plt.xlabel('Time (s)')
    plt.ylabel('Event count')
    plt.title('Event timestamps distribution')
    plt.tight_layout()
    
    os.makedirs("tmp_scripts", exist_ok=True)
    plt.savefig("tmp_scripts/event_histogram.png")
    plt.close()
    io.close()
except Exception as e:
    with open("tmp_scripts/event_histogram_error.txt", "w") as f:
        f.write(str(e))