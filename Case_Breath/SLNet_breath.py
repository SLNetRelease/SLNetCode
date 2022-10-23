import os,sys,math,torch,scipy,imp
import numpy as np
import scipy.io as scio
import torch.nn as nn
from scipy import signal
from math import pi
import matplotlib.pyplot as plt2

SLNet_HiFi_Filter_trainer = imp.load_source('SLNet_HiFi_Filter_trainer', 'SLNet_HiFi_Filter_trainer.py')
STFT_spectrum = imp.load_source('STFT_spectrum', 'Modules/STFT_spectrum.py')
wifilib = imp.load_source('wifilib', 'Modules/wifilib.py')
Acc_meter_util = imp.load_source('Acc_meter_util', 'Acc_meter_util.py')
from SLNet_HiFi_Filter_trainer import SLNet_HiFi_Filter, generate_blur_matrix_complex, complex_array_to_bichannel_float_tensor, bichannel_float_tensor_to_complex_array
from STFT_spectrum import STFT
from wifilib import csi_get_all
from Acc_meter_util import get_acc_meter_spec

csi_down_sample_ratio = int(1000/10)    # 1000Hz => 10Hz
# acc_down_sample_ratio = int(100/10)     # 100Hz => 10Hz
W = 251     # 25.1 seconds for FFT
wind_type = 'gaussian'
file_path_acc_meter_p1 = 'xxx.csv'
file_path_acc_meter_p2 = 'xxx.csv'
file_path_csi = "xxx.dat"
# ------------
str_model_name = 'xxx.pt'

def preprocess_csi(file_path_csi):
    print('Loading data ...')
    csi, timestamp = csi_get_all(file_path_csi)      # csi_np: [T,1,3,30]
    [NTx, NRx] = csi.shape[1:3]
    print('Loaded csi data: ' + str(csi.shape))

    # Downsample
    csi = csi[0::csi_down_sample_ratio,:,:,:]

    # Remove phase
    csi = abs(csi)

    # Reshape [T,1,3,30]=>[90,T]
    csi = csi.reshape((-1,NTx*NRx*30)).swapaxes(1,0)

    # Bandpass filter
    [lu, ld] = signal.butter(5, 0.6, 'low', fs=1000/csi_down_sample_ratio, analog=False)
    [hu, hd] = signal.butter(2, 0.1, 'high', fs=1000/csi_down_sample_ratio, analog=False)
    csi = signal.lfilter(lu, ld, csi, axis=-1)
    csi = signal.lfilter(hu, hd, csi, axis=-1)

    return csi

def csi_to_spec(file_path_csi):
    global W
    # Preprocess CSI
    csi = preprocess_csi(file_path_csi) # [90,T]

    # STFT  (HiFiFilter must use 121/1000 freq bins)
    f_bins, spec = STFT(csi, fs=int(1000/csi_down_sample_ratio), stride=1,\
        wind_wid=W, dft_wid=1000, window_type='gaussian')           # [90,1000,T]j

    # Crop
    spec_crop = np.concatenate((spec[:,:61], spec[:,-60:]), axis=1) # [90,1000,T]j=>[90,121,T]j
    f_bins = np.concatenate((f_bins[:61], f_bins[-60:]), axis=0)

    # Unwrap
    _shift = np.argmax(f_bins)
    spec_crop_unwrap = np.roll(spec_crop, shift=_shift, axis=1)     # [90,121,T]j
    f_bins = np.roll(f_bins, shift=_shift, axis=0)

    # Normalize
    spec_crop_unwrap_norm = normalize_data(spec_crop_unwrap)        # [90,121,T]j
    if np.sum(np.isnan(spec_crop_unwrap_norm)):
        print('>>>>>>>>> NaN detected!')
    return f_bins, spec_crop_unwrap_norm

def normalize_data(data_1):
    # max=1
    # data(ndarray.complex)=>data_norm(ndarray.complex): [C,F,T]j=>[C,F,T]j
    data_1_abs = abs(data_1)
    data_1_max = data_1_abs.max(axis=(1,2),keepdims=True)     # [C,F,T]j=>[C,1,1]j
    data_1_max_rep = np.tile(data_1_max,(1,data_1_abs.shape[1],data_1_abs.shape[2]))    # [C,1,1]j=>[C,F,T]j
    data_1_norm = data_1 / data_1_max_rep
    return  data_1_norm

# ======================== Start Here ========================
if __name__ == "__main__":

    # ========================================== Wi-Fi results
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Load trained model
    print('Loading model...')
    model = torch.load(str_model_name)

    print('Testing model...')
    model.eval()
    with torch.no_grad():
        f_bins, spec_raw = csi_to_spec(file_path_csi)               # [121,], [90,121,T]j
        BPM_bins_HiFi = f_bins * 60

        # Apply SEN
        x_tilde = complex_array_to_bichannel_float_tensor(spec_raw) # [90,121,T]j=>[90,121,2,T]j
        x_tilde = x_tilde.permute(0,3,2,1)                          # [90,121,2,T]=>[90,T,2,121]j
        DFS_WiFi = model(x_tilde.cuda()).cpu()                      # [90,T,2,121]j
        DFS_WiFi = bichannel_float_tensor_to_complex_array(DFS_WiFi)# [90,T,2,121]j=>[90,T,121]j
        DFS_WiFi = np.transpose(DFS_WiFi,(0,2,1))                   # [90,T,121]j=>[90,121,T]j
        DFS_WiFi = np.abs(DFS_WiFi)                                 # [90,121,T]j=>[90,121,T]
        DFS_WiFi = np.mean(DFS_WiFi, axis=0, keepdims=False)        # [90,121,T]=>[121,T] (10Hz)

        # Save raw SPEC
        x_tilde = bichannel_float_tensor_to_complex_array(x_tilde)  # [90,T,2,121]j=>[90,T,121]j
        x_tilde = np.transpose(x_tilde,(0,2,1))                     # [90,T,121]j=>[90,121,T]j
        x_tilde = np.abs(x_tilde)                                   # [90,121,T]j=>[90,121,T]
        x_tilde = np.mean(x_tilde, axis=0, keepdims=False)          # [90,121,T]=>[121,T] (10Hz)

        # Save HiFi&raw SPEC
        scio.savemat('SLNet_Breath_BPM_bins_HiFi_W' + str(W) + '.mat', {'BPM_bins_HiFi':BPM_bins_HiFi})
        scio.savemat('SLNet_Breath_DFS_HiFi_W' + str(W) + '.mat', {'DFS_WiFi':DFS_WiFi})
        scio.savemat('SLNet_Breath_DFS_raw_W' + str(W) + '.mat', {'DFS_raw':x_tilde})
    
    # ========================================== Acc-meter results
    f_bins, DFS_Acc_meter_p1 = get_acc_meter_spec(file_path_acc_meter_p1)   # [F2,T] (100Hz)
    f_bins, DFS_Acc_meter_p2 = get_acc_meter_spec(file_path_acc_meter_p2)   # [F2,T] (100Hz)
    BPM_bins_Acc_meter = f_bins * 60

    scio.savemat('SLNet_Breath_BPM_bins_Acc_meter.mat', {'BPM_bins_Acc_meter':BPM_bins_Acc_meter})
    scio.savemat('SLNet_Breath_DFS_Acc_meter_p1.mat', {'DFS_Acc_meter_p1':DFS_Acc_meter_p1})
    scio.savemat('SLNet_Breath_DFS_Acc_meter_p2.mat', {'DFS_Acc_meter_p2':DFS_Acc_meter_p2})