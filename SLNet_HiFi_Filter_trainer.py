import os,sys,math,scipy,imp
import numpy as np
import scipy.io as scio

import torch, torchvision
import torch.nn as nn
from scipy import signal
from math import sqrt,log,pi
from torch.fft import fft,ifft
from torch.nn.functional import relu, softmax, cross_entropy
from torch import sigmoid,tanh
from torch.nn import MSELoss as MSE

custom_layers_torch = imp.load_source('custom_layers_torch', 'Modules/custom_layers_torch.py')
from custom_layers_torch import m_cconv3d, m_Linear, m_Filtering, m_pconv3d


# Definition
use_existing_model = False
wind_len = 61
wind_type = 'gaussian'
n_max_freq_component = 3
AWGN_amp = 0.01
str_modelname_prefix = 'xxx/' + wind_type + '/' + wind_type + '_W' +\
    str(wind_len) + '/SLNet_HiFi_Filter_' + wind_type + '_W' + str(wind_len)
str_model_name_pretrained = str_modelname_prefix + '_E1000.pt'
feature_len = 121
padded_len = 1000
crop_len = feature_len
blur_matrix_left = []

# Hyperparameters
str_optz = 'RMSprop'
n_begin_epoch = 1       # <====
n_epoch = 2500          # <====
n_itr_per_epoch = 1     # <====
n_batch_size = 64       # <====
n_test_size = 200
f_learning_rate = 0.001

def complex_array_to_bichannel_float_tensor(x):
    # x: (ndarray.complex128) [@,*,H]
    # ret: (tensor.float32) [@,*,2,H]
    x = x.astype('complex64')
    x_real = x.real     # [@,*,H]
    x_imag = x.imag     # [@,*,H]
    ret = np.stack((x_real,x_imag), axis=-2)    # [@,*,H]=>[@,*,2,H]
    ret = torch.tensor(ret)
    return ret

def bichannel_float_tensor_to_complex_array(x):
    # x: (tensor.float32) [@,*,2,H]
    # ret: (ndarray.complex64) [@,*,H]
    x = x.numpy()
    x = np.moveaxis(x,-2,0)  # [@,*,2,H]=>[2,@,*,H]
    x_real = x[0,:]
    x_imag = x[1,:]
    ret = x_real + 1j*x_imag
    return ret

def generate_blur_matrix_complex(wind_type, wind_len=251, padded_len=1000, crop_len=121):
    # Parameters offloading
    fs = 1000
    n_f_bins = crop_len
    f_high = int(n_f_bins/2)
    f_low = -1 * f_high
    init_phase = 0

    # Carrier
    t_ = np.arange(0,wind_len).reshape(1,wind_len)/fs       # [1,wind_len] (0~wind_len/fs seconds)
    freq = np.arange(f_low,f_high+1,1).reshape(n_f_bins,1)  # [n_f_bins,1] (f_low~f_high Hz)
    phase = 2 * pi * freq * t_ + init_phase                 # [n_f_bins,wind_len]
    signal = np.exp(1j*phase)                               # [n_f_bins,wind_len]~[121,251]

    # Windowing
    if wind_type == 'gaussian':
        window = scipy.signal.windows.gaussian(wind_len, (wind_len-1)/sqrt(8*log(200)), sym=True)   # [wind_len,]
    else:
        window = scipy.signal.get_window(wind_type, wind_len)
    sig_wind = signal * window       # [n_f_bins,wind_len]*[wind_len,]=[n_f_bins,wind_len]~[121,251]

    # Pad/FFT
    sig_wind_pad = np.concatenate((sig_wind, np.zeros((n_f_bins,padded_len-wind_len))),axis=1)  # [n_f_bins,wind_len]=>[n_f_bins,padded_len]
    sig_wind_pad_fft = np.fft.fft(sig_wind_pad, axis=-1)    # [n_f_bins,padded_len]~[121,1000]

    # Crop
    n_freq_pos = f_high + 1
    n_freq_neg = abs(f_low)
    sig_wind_pad_fft_crop = np.concatenate((sig_wind_pad_fft[:,:n_freq_pos],\
        sig_wind_pad_fft[:,-1*n_freq_neg:]), axis=1)      # [n_f_bins,crop_len]~[121,121]

    # Unwrap
    n_shift = n_freq_neg
    sig_wind_pad_fft_crop_unwrap = np.roll(sig_wind_pad_fft_crop, shift=n_shift, axis=1) # [n_f_bins,crop_len]~[121,121]

    # Norm (amp_max=1)
    _sig_amp = np.abs(sig_wind_pad_fft_crop_unwrap)
    _sig_ang = np.angle(sig_wind_pad_fft_crop_unwrap)
    _max = np.tile(_sig_amp.max(axis=1,keepdims=True), (1,crop_len))
    _min = np.tile(_sig_amp.min(axis=1,keepdims=True), (1,crop_len))
    _sig_amp_norm = _sig_amp / _max
    sig_wind_pad_fft_crop_unwrap_norm = _sig_amp_norm * np.exp(1j*_sig_ang)

    # Return
    ret = sig_wind_pad_fft_crop_unwrap_norm

    return ret

def syn_one_batch_complex(blur_matrix_right, num_carrier_cand, feature_len, n_batch):

    # Syn. x [@,feature_len]
    x = np.zeros((n_batch, feature_len))*np.exp(1j*0)
    for i in range(n_batch):
        a = np.random.randint(0,len(num_carrier_cand),1)
        num_carrier = num_carrier_cand[int(np.random.randint(0,len(num_carrier_cand),1))]
        idx_carrier = np.random.permutation(feature_len)[:num_carrier]
        x[i,idx_carrier] = np.random.rand(1,num_carrier) * np.exp(1j*( 2*pi*np.random.rand(1,num_carrier) - pi ))

    # Syn. x_blur [@,feature_len]
    x_blur = x @ blur_matrix_right

    # Syn. x_tilde [@,feature_len]
    x_tilde = x_blur + 2*AWGN_amp*(np.random.random(x_blur.shape)-0.5) *\
        np.exp(1j*( 2*pi*np.random.random(x_blur.shape) - pi ))

    return x, x_blur, x_tilde

def loss_function(x, y):
    # x,y: [@,*,2,H]

    x = torch.linalg.norm(x,dim=-2) # [@,*,2,H]=>[@,*,H]
    y = torch.linalg.norm(y,dim=-2) # [@,*,2,H]=>[@,*,H]

    # MSE loss for Amp
    loss_recon = MSE(reduction='mean')(x, y)
    # TODO: Other loss (take phase into consideration)
    return loss_recon 

class SLNet_HiFi_Filter(nn.Module):
    def __init__(self, feature_len):
        super(SLNet_HiFi_Filter, self).__init__()
        self.feature_len = feature_len

        # MLP for Regression
        self.fc_1 = m_Linear(feature_len, feature_len)
        self.fc_2 = m_Linear(feature_len, feature_len)
        self.fc_3 = m_Linear(feature_len, feature_len)
        self.fc_4 = m_Linear(feature_len, feature_len)
        self.fc_out = m_Linear(feature_len, feature_len)

    def forward(self, x):
        h = x   # (@,*,2,H)

        h = tanh(self.fc_1(h))          # (@,*,2,H)=>(@,*,2,H)
        h = tanh(self.fc_2(h))          # (@,*,2,H)=>(@,*,2,H)
        h = tanh(self.fc_3(h))          # (@,*,2,H)=>(@,*,2,H)
        h = tanh(self.fc_4(h))          # (@,*,2,H)=>(@,*,2,H)
        output = tanh(self.fc_out(h))   # (@,*,2,H)=>(@,*,2,H)

        return output

def train(model, blur_matrix_right, feature_len, n_epoch, n_itr_per_epoch, n_batch_size, optimizer):
    valid_loss = []
    for i_epoch in range(n_begin_epoch, n_epoch+1):
        model.train()
        total_loss_this_epoch = 0
        for i_itr in range(n_itr_per_epoch):
            x, _, x_tilde = syn_one_batch_complex(blur_matrix_right=blur_matrix_right, num_carrier_cand=[1,2,3], feature_len=feature_len, n_batch=n_batch_size)
            x = complex_array_to_bichannel_float_tensor(x)
            x_tilde = complex_array_to_bichannel_float_tensor(x_tilde)
            x = x.cuda()
            x_tilde = x_tilde.cuda()

            optimizer.zero_grad()
            y = model(x_tilde)
            loss = loss_function(x, y)
            loss.backward()
            optimizer.step()
            
            total_loss_this_epoch += loss.item()
            
            if i_itr % 1 == 0:
                print('--------> Epoch: {}/{} loss: {:.4f} [itr: {}/{}]'.format(
                    i_epoch+1, n_epoch, loss.item() / n_batch_size, i_itr+1, n_itr_per_epoch), end='\r')
        
        # Validate
        model.eval()
        x, _, x_tilde = syn_one_batch_complex(blur_matrix_right=blur_matrix_right, num_carrier_cand=[1,2,3], feature_len=feature_len, n_batch=n_batch_size)
        x = complex_array_to_bichannel_float_tensor(x)
        x_tilde = complex_array_to_bichannel_float_tensor(x_tilde)
        x = x.cuda()
        x_tilde = x_tilde.cuda()
        y = model(x_tilde)
        total_valid_loss = loss_function(x, y)
        valid_loss.append(total_valid_loss.item())
        print('========> Epoch: {}/{} Loss: {:.4f}'.format(i_epoch+1, n_epoch, total_valid_loss) + ' ' + wind_type + '_' + str(wind_len) + ' '*20)

        if i_epoch % 500 == 0:
            torch.save(model, str_modelname_prefix+'_E'+str(i_epoch)+'.pt')
    return valid_loss

def test(model, blur_matrix_right, feature_len, n_test_size):
    model.eval()
    with torch.no_grad():
        x, x_blur, x_tilde = syn_one_batch_complex(blur_matrix_right=blur_matrix_right, num_carrier_cand=[1,2,3], feature_len=feature_len, n_batch=n_test_size)
        x_tilde = complex_array_to_bichannel_float_tensor(x_tilde)
        y = model(x_tilde.cuda())

        # Save x,y,x_blur,y_blur to file
        y = bichannel_float_tensor_to_complex_array(y.cpu())
        x_tilde = bichannel_float_tensor_to_complex_array(x_tilde)

# ======================== Start Here ========================
if __name__ == "__main__":
    if len(sys.argv) < 1:
        print('Please specify which GPU to use ...')
        exit(0)
    if (sys.argv[1] == '1' or sys.argv[1] == '0'):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    else:
        print('Wrong GPU number, 0 or 1 supported!')
        exit(0)

    # Generate blur matrix
    blur_matrix_right = generate_blur_matrix_complex(wind_type=wind_type, wind_len=wind_len, padded_len=padded_len, crop_len=crop_len)
    scio.savemat('SLNet_HiFi_A_' + wind_type + '_W' + str(wind_len) + '.mat', {'A':blur_matrix_right})

    # Load or fabricate model
    if use_existing_model:
        print('Model loading...')
        model = torch.load(str_model_name_pretrained)
    else:
        print('Model building...')
        model = SLNet_HiFi_Filter(feature_len=feature_len)
        torch.save(model, str_modelname_prefix+'_E0.pt')
        model.cuda()

    # Train model
    if str_optz == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=f_learning_rate)
    elif str_optz == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=f_learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=f_learning_rate)
    print('Model training...')
    valid_loss = train(model=model, blur_matrix_right=blur_matrix_right, feature_len=feature_len, n_epoch=n_epoch, n_itr_per_epoch=n_itr_per_epoch, n_batch_size=n_batch_size, optimizer=optimizer)
    scio.savemat('SLNet_HiFiFilter_valid_loss.mat', {'valid_loss':valid_loss})

    # Test model
    test(model=model, blur_matrix_right=blur_matrix_right, feature_len=feature_len, n_test_size=n_test_size)