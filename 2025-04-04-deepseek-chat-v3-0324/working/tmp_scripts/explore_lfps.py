# Script to explore LFP data from NWB file
# Shows sample traces from 3 channels and saves plot to file

import matplotlib.pyplot as plt
import pynwb
import lindi

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/000673/assets/f3a72b16-6e70-4adf-8384-28224ce212a8/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get LFP data
lfps = nwb.acquisition["LFPs"]
data = lfps.data[:]  # Get all LFP data (shape: [556712, 55])
rate = lfps.rate  # 400 Hz

# Plot sample traces from 3 channels
plt.figure(figsize=(12, 6))
for i, ch in enumerate([0, 10, 20]):  # Sample channels
    plt.plot(data[:1000, ch] + i*1000, label=f'Channel {ch}')
    
plt.title('Sample LFP Traces (First 1000 samples)')
plt.xlabel('Sample number')
plt.ylabel('Voltage (uV) with offset')
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig('tmp_scripts/lfp_traces.png')
plt.close()