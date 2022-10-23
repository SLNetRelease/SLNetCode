import imp
import numpy as np
from scipy import interpolate
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy import signal
from math import pi
STFT_spectrum = imp.load_source('STFT_spectrum', 'Modules/STFT_spectrum.py')
from STFT_spectrum import STFT

# Load Acc data from csv data
# Output: acc_xyz:[1,T], timestamp:[T,]
def load_acc_meter(file_name):
    acc_xyz = []
    timestamp = []
    with open(file_name, newline='') as csvfile:
        for _lines in csvfile.readlines()[1:]:
            _lines = _lines.split(",")
            acc_xyz.append([float(_lines[1]),float(_lines[2]),float(_lines[3])])
            timestamp.append(float(_lines[0]))
    timestamp = np.array(timestamp)                         # [T,]
    acc_xyz = np.array(acc_xyz)                             # [T,3]
    acc_xyz = np.sqrt(np.sum(acc_xyz * acc_xyz, axis=1))    # [T,3]=>[T,]
    acc_xyz = np.expand_dims(acc_xyz, axis=0)               # [T,]=>[1,T]

    # Bandpass filter
    [lu, ld] = signal.butter(5, 0.6, 'low', fs=100, analog=False)   # Up:  0.6Hz=36BPM
    [hu, hd] = signal.butter(2, 0.1, 'high', fs=100, analog=False)  # Low: 0.1Hz=6BPM
    acc_xyz = signal.lfilter(lu, ld, acc_xyz, axis=-1)
    acc_xyz = signal.lfilter(hu, hd, acc_xyz, axis=-1)      # [1,T]

    return acc_xyz, timestamp

# Input: acc_data:[1,T]
# Output: Spec: [1,F,T]
def process_acc_meter(acc_data):
    fs = 100

    # STFT
    f_bins, spec = STFT(acc_data, fs=fs, stride=1, wind_wid=60*fs,\
        dft_wid=60*fs, window_type='gaussian') # [T,],[1,3000,T]j

    # Crop
    spec = np.squeeze(spec)
    spec_crop = np.concatenate((spec[:37,:],spec[-36:,:]), axis=0)  # [3000,T]j=>[25,T]j
    f_bins = np.concatenate((f_bins[:37],f_bins[-36:]), axis=0)     # [3000,]=>[25,]

    # Unwrap
    _shift = np.argmax(f_bins)
    spec_crop_unwrap = np.roll(spec_crop, shift=_shift, axis=0)     # [25,T]j
    f_bins = np.roll(f_bins, shift=_shift, axis=0)

    return f_bins, spec_crop_unwrap

def get_acc_meter_spec(file_name):
    acc_xyz, timestamp = load_acc_meter(file_name)          # [T,], [T,]
    f_bins, DFS_Acc_meter = process_acc_meter(acc_xyz)      # [T,]=>[1,F,T]
    
    return f_bins, DFS_Acc_meter

# ======================== Start Here ========================
if __name__ == "__main__":
    f_bins, DFS_Acc_meter = get_acc_meter_spec('xxx.csv')     # [F,T]
    BPM_bins = f_bins * 60

    scio.savemat('SLNet_Breath_BPM_bins_Acc_meter.mat', {'BPM_bins_Acc_meter':BPM_bins})
    scio.savemat('SLNet_Breath_DFS_Acc_meter.mat', {'DFS_Acc_meter':DFS_Acc_meter})