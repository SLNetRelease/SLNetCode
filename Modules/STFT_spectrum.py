import os,sys
import scipy
import numpy as np
from scipy import signal
from math import sqrt,log
import time

def STFT(signal, fs=1, stride=1, wind_wid=5, dft_wid=5, window_type='gaussian'):
    assert dft_wid >= wind_wid and wind_wid > 0 and stride <= wind_wid and stride > 0\
        and isinstance(stride, int) and isinstance(wind_wid, int) and isinstance(dft_wid, int)\
        and isinstance(fs, int) and fs > 0

    # Construct STFT window
    if window_type == 'gaussian':
        window = scipy.signal.windows.gaussian(wind_wid, (wind_wid-1)/sqrt(8*log(200)), sym=True)
    elif window_type == 'rect':
        window = np.ones((wind_wid,))
    else:
        window = scipy.signal.get_window(window_type, wind_wid)

    f_bins, t_bins, stft_spectrum = scipy.signal.stft(x=signal, fs=fs, window=window, nperseg=wind_wid, noverlap=wind_wid-stride, nfft=dft_wid,\
        axis=-1, detrend=False, return_onesided=False, boundary='zeros', padded=True)
    
    return f_bins, stft_spectrum
