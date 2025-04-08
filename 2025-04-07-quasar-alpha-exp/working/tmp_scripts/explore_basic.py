"""
Exploratory script:
- Load NWB file metadata and structure using lindi + pynwb
- Print summary info
- Plot a 5-second segment of LFPs averaged across electrodes
- Plot histogram of event codes
- Raster plot of spike times for first 5 units during first 10 seconds

PNG plots saved to tmp_scripts; NO plt.show()
"""

import matplotlib.pyplot as plt
import numpy as np
import lindi
import pynwb

# Load the NWB file via LINDI
print("Loading NWB file...")
url = "https://lindi.neurosift.org/dandi/dandisets/000673/assets/95406971-26ad-4894-917b-713ed7625349/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print('Session:', nwb.session_description)
print('Subject:', nwb.subject.subject_id, 'Sex:', nwb.subject.sex, 'Age:', nwb.subject.age)
print('Experimenter:', nwb.experimenter)
print('Institution:', nwb.institution)

## Trial info
trials = nwb.intervals['trials']
print("Number of trials:", len(trials['id']))

## Events TTL markers
events = nwb.acquisition['events']
event_data = events.data[:]
event_times = events.timestamps[:]

## Plot event histogram
plt.figure()
plt.hist(event_data, bins=np.arange(event_data.min()-0.5, event_data.max()+1.5, 1))
plt.xlabel('Event code')
plt.ylabel('Count')
plt.title('Histogram of event TTL codes')
plt.savefig('tmp_scripts/event_histogram.png')

## LFP: plot average trace over 5 seconds (first 2000 samples at 400 Hz)
LFPs = nwb.acquisition['LFPs']
rate = LFPs.rate
num_samples = int(5 * rate)
lfp_data = LFPs.data[:num_samples, :]   # shape (num_samples, num_channels)
avg_lfp = lfp_data.mean(axis=1)

time_vec = np.arange(num_samples) / rate
plt.figure()
plt.plot(time_vec, avg_lfp)
plt.xlabel('Time (s)')
plt.ylabel('Mean LFP across channels')
plt.title('Average LFP over first 5 seconds')
plt.savefig('tmp_scripts/lfp_avg_trace.png')

## Units raster for first 5 units in the first 10 sec
units = nwb.units
unit_ids = units['id'].data[:]
num_units = min(5, len(unit_ids))
plt.figure(figsize=(8, num_units*0.7))
for i in range(num_units):
    spk_times = units['spike_times'][i]
    mask = (spk_times >= 0) & (spk_times <= 10)
    plt.vlines(spk_times[mask], i + 0.5, i + 1.5)
plt.xlabel('Time (s)')
plt.ylabel('Neuron #')
plt.yticks(np.arange(1, num_units+1), [f"ID {unit_ids[i]}" for i in range(num_units)])
plt.title('Spike raster of first 5 units (first 10 seconds)')
plt.savefig('tmp_scripts/unit_raster.png')