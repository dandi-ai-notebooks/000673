# %% [markdown]
# # Data exploration notebook: DANDI 000673
# 
# This notebook is **AI-generated via dandi-notebook-gen** to assist initial exploration of this Dandiset:
# 
# **"Data for: Control of working memory by phase–amplitude coupling of human hippocampal neurons"**
# 
# Data consists of simultaneous medial temporal and frontal single-unit and LFP recordings from epilepsy patients performing working memory tasks (Sternberg paradigm), with multi-electrode depth probes.  
# 
# ⚠️ **Note:** This notebook **has not been fully verified**. Interpret all examples cautiously. Do **not** draw scientific conclusions without further data validation and appropriate statistical analyses.
# 
# The data and code here are intended only as illustrative starting points.

# %% [markdown]
# ## Dandiset Metadata
# - Keywords: cognitive neuroscience, working memory, neurophysiology, phase-amplitude coupling
# - Measurement types: multi-electrode extracellular, single-unit spikes
# - Sighted units: 41; electrodes: ~69-73 per file; trials: 140+
# - More code at: https://github.com/rutishauserlab/SBCAT-release-NWB

# %% [markdown]
# ## List all assets (NWB files) in this Dandiset

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("000673")
assets = list(dandiset.get_assets())
print(f"Found {len(assets)} assets")
for asset in assets[:5]:
    print(asset.path)

# %% [markdown]
# Example analyses below all use **subject 1, session 2** file for demo purposes:

# %%
import pynwb
import lindi

url = "https://lindi.neurosift.org/dandi/dandisets/000673/assets/95406971-26ad-4894-917b-713ed7625349/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print(nwb.session_description)
print('Trials:', len(nwb.intervals['trials']['id']))
print('Units:', len(nwb.units['id'].data[:]))

# %% [markdown]
# ## Behavioral event TTL markers

# %%
import matplotlib.pyplot as plt
import numpy as np

events = nwb.acquisition['events']
event_data = events.data[:]
event_times = events.timestamps[:]

plt.figure()
plt.hist(event_data, bins=np.arange(event_data.min()-0.5, event_data.max()+1.5, 1))
plt.xlabel('Event code')
plt.ylabel('Count')
plt.title('TTL event codes histogram')
plt.savefig('tmp_scripts/event_histogram.png')
plt.show()

# %% [markdown]
# This histogram shows most frequent TTL codes around 1, 2, 7, 11 — consistent with task trial markers. Caution should be used when interpreting sparse or missing codes.

# %% [markdown]
# ## Example LFP data segment (avg across electrodes)

# %%
LFPs = nwb.acquisition['LFPs']
rate = LFPs.rate  # 400 Hz downsampled, spike-removed
num_seconds = 5
num_samples = int(rate * num_seconds)
lfp_data = LFPs.data[:num_samples, :]
avg_lfp = lfp_data.mean(axis=1)
time_vec = np.arange(num_samples) / rate
plt.figure()
plt.plot(time_vec, avg_lfp)
plt.xlabel('Time (s)')
plt.ylabel('Mean LFP')
plt.title(f'Average LFP first {num_seconds} seconds')
plt.savefig('tmp_scripts/lfp_avg_trace.png')
plt.show()

# %% [markdown]
# Average LFP shows a transient ~1.2s in, then moderate oscillations suitable for demonstrations.

# %% [markdown]
# ## Example spike raster of 5 units

# %%
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
plt.title('Spike raster for 5 units (first 10 s)')
plt.savefig('tmp_scripts/unit_raster.png')
plt.show()

# %% [markdown]
# The raster shows diversity in spiking rates and patterns across example units.

# %% [markdown]
# ## Loading trial table info and behavioral responses
# - The NWB file contains detailed trial timings, stimulus IDs, probe correctness, etc.
# - Below: preview columns, display first few entries.

# %%
trials = nwb.intervals['trials']
print("Trial columns:", list(trials.colnames))
for k in ['loads', 'probe_in_out', 'response_accuracy']:
    vals = trials[k][:]
    print(f"{k} (first 10 trials):", vals[:10])

# %% [markdown]
# ## Next steps for user
# - This series of examples demonstrate **basic** NWB data loading, visualization
# - Extend by:
#   - Selecting task periods using TTLs or trial timings
#   - Computing spike counts or firing rates aligned to events
#   - Spectral analysis of LFP (use smaller time windows)
#   - Cross-regional comparisons
#   - Careful statistical analyses and controls
# 
# For published code and more advanced workflows, consult: https://github.com/rutishauserlab/SBCAT-release-NWB
# 
# *Again:* This notebook is **for illustration only and not a substitute for in-depth, tailored analysis.