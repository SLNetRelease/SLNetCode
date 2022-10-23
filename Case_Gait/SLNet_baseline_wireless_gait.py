import os,sys,math,imp
import scipy
import numpy as np
import scipy.io as scio

import torch, torchvision
import torch.nn as nn
from torch.nn.functional import relu, softmax, cross_entropy
from torch import sigmoid,tanh

custom_layers_torch = imp.load_source('custom_layers_torch', 'Modules/custom_layers_torch.py')
SLNet_HiFi_Filter_trainer = imp.load_source('SLNet_HiFi_Filter_trainer', 'SLNet_HiFi_Filter_trainer.py')
from custom_layers_torch import m_cconv3d, m_Linear, m_Filtering, m_pconv3d
from SLNet_HiFi_Filter_trainer import SLNet_HiFi_Filter, complex_array_to_bichannel_float_tensor, bichannel_float_tensor_to_complex_array
from thop import profile


# Definition
use_adversarial = True
use_existing_SLNet_model = False    # Note: NOT HiFiFilter model
W = 251                             # <====
wind_type = 'gaussian'              # <====
downsamp_ratio_for_stft = 10        # 100Hz / 10 = 10Hz        # <====
data_dir = 'xxx/'
epoch_filter = 5000
str_SLNet_name = 'SLNet_gait_torch.pt'
str_HiFiFilter_name = 'xxx.pt'
mat_file_key = 'SPEC_W' + str(W)
# --------------------------
ID_LIST = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10', 'id11']
N_PEOPLE = len(ID_LIST)
T_MAX = 0
frac_for_valid = 0.1
frac_for_test = 0.05

# Hyperparameters
str_optz = 'RMSprop'
f_learning_rate = 0.001
n_batch_size = 128
n_epoch = 100


def normalize_data(data_1):
    # max=1 for each [F,T] spectrum
    # data(ndarray.complex)=>data_norm(ndarray.complex): [6,121,T]=>[6,121,T]
    data_1_abs = abs(data_1)
    data_1_max = data_1_abs.max(axis=(1,2),keepdims=True)     # [6,121,T]=>[6,1,1]
    data_1_max_rep = np.tile(data_1_max,(1,data_1_abs.shape[1],data_1_abs.shape[2]))    # [6,1,1]=>[6,121,T]
    data_1_norm = data_1 / data_1_max_rep   # [6,121,T]
    return  data_1_norm

def zero_padding(data, T_MAX):
    # data(list)=>data_pad(ndarray): ([6,121,t1/t2/...],...)=>[6,121,T_MAX]
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[-1]
        data_pad.append(np.pad(data[i], ((0,0),(0,0),(T_MAX - t,0)), 'constant', constant_values = 0).tolist())   # Front padding
    res = np.array(data_pad)
    return res

def complex_array_to_2_channel_float_array(data_complex):
    # data_complex(complex128/float64)=>data_float: [N,6,121,T_MAX]=>[N,2,6,121,T_MAX]
    data_complex = data_complex.astype('complex64')
    data_real = data_complex.real
    data_imag = data_complex.imag
    data_2_channel_float = np.stack((data_real, data_imag), axis=1)
    return data_2_channel_float

def load_data_to_array(path_to_data):
    # Need customization
    # data: [N,6,121,T_MAX]
    # label: [N,]
    global T_MAX
    print('Loading data from ' + str(path_to_data))

    # Load data
    data = []
    label = []
    domain = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:

            file_path = os.path.join(data_root,data_file_name)
            try:
                # Label embedding
                label_1_name = data_file_name.split('-')[0]
                domain_1 = int(data_file_name.split('-')[1])
                if label_1_name in ID_LIST:
                    label_1 = ID_LIST.index(label_1_name)+1
                else:
                    continue

                # Load raw data
                data_1 = abs(scio.loadmat(file_path)[mat_file_key]) # [6,121,T]

                # Skip rx is not equal to 6
                if data_1.shape[0] is not 6:
                    print('Skipping ' + str(data_file_name) + ', Rx not 6')
                    continue

                # Downsample
                data_1 = data_1[:,:,0::downsamp_ratio_for_stft]     # [6,121,T]=>[6,121,t]

                # Skip nan
                if np.sum(np.isnan(data_1)):
                    print('Skipping ' + str(data_file_name))
                    continue

                # Normalization
                data_normed_1 = normalize_data(data_1)  # [6,121,t]

                # Update T_MAX
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]                
            except Exception as e:
                print(str(e))
                continue

            # Save List
            data.append(data_normed_1.tolist())
            label.append(label_1-1)
            domain.append(domain_1-1)

    # Zero-padding
    data = zero_padding(data, T_MAX)    # ([6,121,t1],...)=>[N,6,121,T]

    # Convert from complex128 to 2_channel_float32
    data = complex_array_to_2_channel_float_array(data)    # [N,6,121,T]=>[N,2,6,121,T]

    # Convert label to ndarray
    label = np.array(label)
    domain = np.array(domain)

    # data(ndarray): [N,2,6,121,T_MAX], label(ndarray): [N,], domain(ndarray): [N,]
    return data, label, domain

def load_data_to_loader(data_dir):
    # Template function
    print('Loading data...')
    # Load numpy => data,label (numpy array)
    data, label, domain = load_data_to_array(data_dir)
    n_class_loaded = np.unique(label).shape[0]
    sample_shape = data[0].shape

    # Numpy => torch.utils.data.TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.tensor(data,dtype=torch.float32), torch.tensor(label,dtype=torch.int64),\
                                             torch.tensor(domain,dtype=torch.int64)) # Note: float32
    sample_dtype = dataset[0][0].dtype

    # Split train/test dataset
    sample_count = dataset.__len__()
    valid_sample_count = int(sample_count*frac_for_valid)
    test_sample_count = int(sample_count*frac_for_test)
    train_sample_count = sample_count - test_sample_count - valid_sample_count
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_sample_count, valid_sample_count, test_sample_count])

    # torch.utils.data.TensorDataset => torch.utils.data.DataLoader
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=n_batch_size, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=valid_sample_count, shuffle=False)
    test_data_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_sample_count, shuffle=False)

    print('Loaded {} samples, split {}/{}/{} for train/valid/test, data shape: {}, data dtype: {}, class num (loaded/expected): {}/{}'.format(\
        sample_count,train_sample_count,valid_sample_count,test_sample_count,sample_shape,sample_dtype, n_class_loaded, N_PEOPLE))
    return train_data_loader, valid_data_loader, test_data_loader, sample_shape

def loss_function(label_batch, label_pred):
    # Need customization
    # label_batch:[@], label_pred:[@,n_class]
    return nn.CrossEntropyLoss()(label_pred, label_batch)   # LogSoftmax + NLLLoss

def loss_function_adv(label_batch, domain_batch, label_pred, domain_pred):
    # Need customization
    # label_batch:[@], label_pred:[@,n_class]
    return nn.CrossEntropyLoss()(label_pred, label_batch) + nn.CrossEntropyLoss()(domain_pred, domain_batch)   # LogSoftmax + NLLLoss

class Baseline_Widar3(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(Baseline_Widar3, self).__init__()
        self.input_shape = input_shape  # [2,6,121,T_MAX]
        self.T_MAX = input_shape[3]
        self.class_num = class_num

        # CNN+RNN
        self.fc_1 = nn.Linear(32*5, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 32)
        self.fc_out_1_1 = nn.Linear((self.T_MAX-10)*128, 128)
        self.fc_out_1_2 = nn.Linear(128, self.class_num)
        self.fc_out_2_1 = nn.Linear((self.T_MAX-10)*128, 128)
        self.fc_out_2_2 = nn.Linear(128, 25)    # 25 domains
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.conv3d_1 = nn.Conv3d(in_channels= 1, out_channels=16, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[1,3,3], stride=[1,1,1])
        self.mpooling3d_1 = nn.MaxPool3d(kernel_size=[1,3,1],stride=[1,3,1])
        self.mpooling3d_2 = nn.MaxPool3d(kernel_size=[1,5,1],stride=[1,5,1])
        self.GRU = nn.GRU(input_size=6*32, hidden_size=128, num_layers=1, bias=True, batch_first=True, dropout = 0.5)

    def forward(self, x):
        h = x   # (@,2,6,121,T_MAX)

        # Baseline: CNN+GRU+Adv
        h = torch.linalg.norm(h,dim=1)          # (@,2,6,121,T_MAX)=>(@,6,121,T_MAX)
        h = torch.unsqueeze(h,dim=1)            # (@,6,121,T_MAX)=>(@,1,6,121,T_MAX)

        h = relu(self.conv3d_1(h))              # (@,1,6,121,T_MAX)=>(@,16,6,117,T_MAX-4)
        h = self.mpooling3d_1(h)                # (@,16,6,117,T_MAX-4)=>(@,16,6,39,T_MAX-4)

        h = relu(self.conv3d_2(h))              # (@,16,6,39,T_MAX-4)=>(@,32,6,35,T_MAX-8)
        h = self.mpooling3d_2(h)                # (@,32,6,35,T_MAX-8)=>(@,32,6,7,T_MAX-8)

        h = relu(self.conv3d_3(h))              # (@,32,6,7,T_MAX-8)=>(@,32,6,5,T_MAX-10)

        h = h.permute(0,2,4,1,3)                # (@,32,6,5,T_MAX-10)=>(@,6,T_MAX-10,32,5)
        h = h.reshape((-1,6,self.T_MAX-10,32*5))# (@,6,T_MAX-10,32,5)=>(@,6,T_MAX-10,32*5)
        h = self.dropout_1(h)
        h = relu(self.fc_1(h))                  # (@,6,T_MAX-10,32*5)=>(@,6,T_MAX-10,128)
        h = self.dropout_2(h)
        h = relu(self.fc_2(h))                  # (@,6,T_MAX-10,128)=>(@,6,T_MAX-10,64)
        h = relu(self.fc_3(h))                  # (@,6,T_MAX-10,64)=>(@,6,T_MAX-10,32)

        h = h.permute(0,2,1,3)                  # (@,6,T_MAX-10,32)=>(@,T_MAX-10,6,32)

        h = h.reshape((-1,self.T_MAX-10,6*32))  # (@,T_MAX-10,6,32)=>(@,T_MAX-10,6*32)
        
        h,_ = self.GRU(h)                       # (@,T_MAX-10,6*32)=>(@,T_MAX-10,128)
        h = h.reshape((-1,(self.T_MAX-10)*128)) # (@,T_MAX-10,128)=>(@,(T_MAX-10)*128)

        # Gesture class
        output_1 = relu(self.fc_out_1_1(h))     # (@,(T_MAX-10)*128)=>(@,128)
        output_1 = self.fc_out_1_2(output_1)    # (@,128)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        return output_1

class Baseline_EI(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(Baseline_EI, self).__init__()
        self.input_shape = input_shape  # [2,6,121,T_MAX]
        self.T_MAX = input_shape[3]
        self.class_num = class_num

        # CNN+FC
        self.fc_1 = nn.Linear(32*5, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 32)
        self.fc_out_1_1 = nn.Linear(6*(self.T_MAX-10)*32, 256)
        self.fc_out_1_2 = nn.Linear(256, self.class_num)
        self.fc_out_2_1 = nn.Linear(6*(self.T_MAX-10)*32, 256)
        self.fc_out_2_2 = nn.Linear(256, 25)    # 25 domains
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.conv3d_1 = nn.Conv3d(in_channels= 1, out_channels=16, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[1,3,3], stride=[1,1,1])
        self.mpooling3d_1 = nn.MaxPool3d(kernel_size=[1,3,1],stride=[1,3,1])
        self.mpooling3d_2 = nn.MaxPool3d(kernel_size=[1,5,1],stride=[1,5,1])

    def forward(self, x):
        h = x   # (@,2,6,121,T_MAX)

        # Baseline: CNN+FC
        h = torch.linalg.norm(h,dim=1)          # (@,2,6,121,T_MAX)=>(@,6,121,T_MAX)
        h = torch.unsqueeze(h,dim=1)            # (@,6,121,T_MAX)=>(@,1,6,121,T_MAX)

        h = relu(self.conv3d_1(h))              # (@,1,6,121,T_MAX)=>(@,16,6,117,T_MAX-4)
        h = self.mpooling3d_1(h)                # (@,16,6,117,T_MAX-4)=>(@,16,6,39,T_MAX-4)

        h = relu(self.conv3d_2(h))              # (@,16,6,39,T_MAX-4)=>(@,32,6,35,T_MAX-8)
        h = self.mpooling3d_2(h)                # (@,32,6,35,T_MAX-8)=>(@,32,6,7,T_MAX-8)

        h = relu(self.conv3d_3(h))              # (@,32,6,7,T_MAX-8)=>(@,32,6,5,T_MAX-10)

        h = h.permute(0,2,4,1,3)                # (@,32,6,5,T_MAX-10)=>(@,6,T_MAX-10,32,5)
        h = h.reshape((-1,6,self.T_MAX-10,32*5))# (@,6,T_MAX-10,32,5)=>(@,6,T_MAX-10,32*5)
        h = self.dropout_1(h)
        h = relu(self.fc_1(h))                  # (@,6,T_MAX-10,32*5)=>(@,6,T_MAX-10,128)
        h = self.dropout_2(h)
        h = relu(self.fc_2(h))                  # (@,6,T_MAX-10,128)=>(@,6,T_MAX-10,64)
        h = relu(self.fc_3(h))                  # (@,6,T_MAX-10,64)=>(@,6,T_MAX-10,32)
        h = h.reshape((-1,6*(self.T_MAX-10)*32)) # (@,6,T_MAX-10,32)=>(@,6*(T_MAX-10)*32)
        h = self.dropout_3(h)

        # Branch-1: Gesture class
        output_1 = relu(self.fc_out_1_1(h))
        output_1 = self.fc_out_1_2(output_1)        # (@,6*(T_MAX-8)*32)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        # Branch-2: Domain
        output_2 = relu(self.fc_out_2_1(h))
        output_2 = self.fc_out_2_2(output_2)        # (@,6*(T_MAX-8)*32)=>(@,n_domain)

        return output_1, output_2

class Baseline_CrossSense(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(Baseline_CrossSense, self).__init__()
        self.input_shape = input_shape  # [2,6,121,T_MAX]
        self.T_MAX = input_shape[3]
        self.n_class = class_num

        # CNN+FC
        self.fc_1 = nn.Linear(6*121, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, 128)
        self.fc_4 = nn.Linear(128, 64)
        self.fc_5 = nn.Linear(64, 32)
        self.fc_6_1 = nn.Linear(32*self.T_MAX, 2048)
        self.fc_6_2 = nn.Linear(2048, 1024)
        self.fc_6 = nn.Linear(1024, 512)
        self.fc_7 = nn.Linear(512, 256)
        self.fc_8 = nn.Linear(256, 128)
        self.fc_9 = nn.Linear(128, 64)
        self.fc_output = nn.Linear(64, self.n_class)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_6 = nn.Dropout(p=0.5)
        self.dropout_7 = nn.Dropout(p=0.5)
        self.dropout_8 = nn.Dropout(p=0.5)
        self.dropout_9 = nn.Dropout(p=0.5)

    def forward(self, x):
        h = x   # (@,2,6,121,T_MAX)

        # Baseline: ANN
        h = torch.linalg.norm(h,dim=1)          # (@,2,6,121,T_MAX)=>(@,6,121,T_MAX)
        h = h.permute(0,3,1,2)                  # (@,6,121,T_MAX)=>(@,T_MAX,6,121)

        h = h.reshape((-1,self.T_MAX,6*121))    # (@,T_MAX,6,121)=>(@,T_MAX,6*121)
        h = self.dropout_1(h)
        h = relu(self.fc_1(h))                  # (@,T_MAX,6*121)=>(@,T_MAX,512)
        h = self.dropout_2(h)
        h = relu(self.fc_2(h))                  # (@,T_MAX,512)=>(@,T_MAX,256)
        h = self.dropout_3(h)
        h = relu(self.fc_3(h))                  # (@,T_MAX,256)=>(@,T_MAX,128)
        h = self.dropout_4(h)
        h = relu(self.fc_4(h))                  # (@,T_MAX,128)=>(@,T_MAX,64)
        h = self.dropout_5(h)
        h = relu(self.fc_5(h))                  # (@,T_MAX,64)=>(@,T_MAX,32)

        h = h.reshape((-1,self.T_MAX*32))       # (@,T_MAX,32)=>(@,T_MAX*32)

        h = relu(self.fc_6_1(h))                # (@,T_MAX*32)=>(@,2048)
        h = relu(self.fc_6_2(h))                # (@,2048)=>(@,1024)

        h = self.dropout_6(h)
        h = relu(self.fc_6(h))                  # (@,1024)=>(@,512)
        h = self.dropout_7(h)
        h = relu(self.fc_7(h))                  # (@,512)=>(@,256)
        h = self.dropout_8(h)
        h = relu(self.fc_8(h))                  # (@,256)=>(@,128)
        h = self.dropout_9(h)
        h = relu(self.fc_9(h))                  # (@,128)=>(@,64)
        output = self.fc_output(h)              # (@,64)=>(@,n_class)

        return output

class Baseline_RFSleep(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(Baseline_RFSleep, self).__init__()
        self.input_shape = input_shape  # [2,6,121,T_MAX]
        self.T_MAX = input_shape[3]
        self.class_num = class_num

        # CNN+RNN
        self.fc_1 = nn.Linear(32*5, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 32)
        self.fc_out_1_1 = nn.Linear((self.T_MAX-10)*128, 128)
        self.fc_out_1_2 = nn.Linear(128, self.class_num)
        self.fc_out_2_1 = nn.Linear((self.T_MAX-10)*128, 128)
        self.fc_out_2_2 = nn.Linear(128, 25)    # 25 domains
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.conv3d_1 = nn.Conv3d(in_channels= 1, out_channels=16, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[1,3,3], stride=[1,1,1])
        self.mpooling3d_1 = nn.MaxPool3d(kernel_size=[1,3,1],stride=[1,3,1])
        self.mpooling3d_2 = nn.MaxPool3d(kernel_size=[1,5,1],stride=[1,5,1])
        self.GRU = nn.GRU(input_size=6*32, hidden_size=128, num_layers=1, bias=True, batch_first=True, dropout = 0.5)

    def forward(self, x):
        h = x   # (@,2,6,121,T_MAX)

        # Baseline: CNN+GRU+Adv
        h = torch.linalg.norm(h,dim=1)          # (@,2,6,121,T_MAX)=>(@,6,121,T_MAX)
        h = torch.unsqueeze(h,dim=1)            # (@,6,121,T_MAX)=>(@,1,6,121,T_MAX)

        h = relu(self.conv3d_1(h))              # (@,1,6,121,T_MAX)=>(@,16,6,117,T_MAX-4)
        h = self.mpooling3d_1(h)                # (@,16,6,117,T_MAX-4)=>(@,16,6,39,T_MAX-4)

        h = relu(self.conv3d_2(h))              # (@,16,6,39,T_MAX-4)=>(@,32,6,35,T_MAX-8)
        h = self.mpooling3d_2(h)                # (@,32,6,35,T_MAX-8)=>(@,32,6,7,T_MAX-8)

        h = relu(self.conv3d_3(h))              # (@,32,6,7,T_MAX-8)=>(@,32,6,5,T_MAX-10)

        h = h.permute(0,2,4,1,3)                # (@,32,6,5,T_MAX-10)=>(@,6,T_MAX-10,32,5)
        h = h.reshape((-1,6,self.T_MAX-10,32*5))# (@,6,T_MAX-10,32,5)=>(@,6,T_MAX-10,32*5)
        h = self.dropout_1(h)
        h = relu(self.fc_1(h))                  # (@,6,T_MAX-10,32*5)=>(@,6,T_MAX-10,128)
        h = self.dropout_2(h)
        h = relu(self.fc_2(h))                  # (@,6,T_MAX-10,128)=>(@,6,T_MAX-10,64)
        h = relu(self.fc_3(h))                  # (@,6,T_MAX-10,64)=>(@,6,T_MAX-10,32)

        h = h.permute(0,2,1,3)                  # (@,6,T_MAX-10,32)=>(@,T_MAX-10,6,32)

        h = h.reshape((-1,self.T_MAX-10,6*32))  # (@,T_MAX-10,6,32)=>(@,T_MAX-10,6*32)
        
        h,_ = self.GRU(h)                       # (@,T_MAX-10,6*32)=>(@,T_MAX-10,128)
        h = h.reshape((-1,(self.T_MAX-10)*128)) # (@,T_MAX-10,128)=>(@,(T_MAX-10)*128)

        # Branch-1: Gesture class
        output_1 = relu(self.fc_out_1_1(h))     # (@,(T_MAX-10)*128)=>(@,128)
        output_1 = self.fc_out_1_2(output_1)    # (@,128)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        # Branch-2: Domain
        output_2 = relu(self.fc_out_2_1(h))     # (@,(T_MAX-10)*128)=>(@,128)
        output_2 = self.fc_out_2_2(output_2)    # (@,128)=>(@,n_domain)

        return output_1, output_2

class Baseline_RFPose(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(Baseline_RFPose, self).__init__()
        self.input_shape = input_shape  # [2,6,121,T_MAX]
        self.T_MAX = input_shape[3]
        self.class_num = class_num

        # CNN+FC
        self.fc_1_1 = nn.Linear(32*5, 128)
        self.fc_2_1 = nn.Linear(128, 64)
        self.fc_3_1 = nn.Linear(64, 32)
        self.fc_1_2 = nn.Linear(32*5, 128)
        self.fc_2_2 = nn.Linear(128, 64)
        self.fc_3_2 = nn.Linear(64, 32)
        self.fc_out_1 = nn.Linear(6*(self.T_MAX-10)*64, 256)
        self.fc_out_2 = nn.Linear(256, self.class_num)
        self.dropout_1_1 = nn.Dropout(p=0.2)
        self.dropout_2_1 = nn.Dropout(p=0.3)
        self.dropout_1_2 = nn.Dropout(p=0.2)
        self.dropout_2_2 = nn.Dropout(p=0.3)
        self.conv3d_1_1 = nn.Conv3d(in_channels= 1, out_channels=16, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_2_1 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_3_1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[1,3,3], stride=[1,1,1])
        self.conv3d_1_2 = nn.Conv3d(in_channels= 1, out_channels=16, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_2_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=[1,5,5], stride=[1,1,1])
        self.conv3d_3_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[1,3,3], stride=[1,1,1])
        self.mpooling3d_1_1 = nn.MaxPool3d(kernel_size=[1,3,1],stride=[1,3,1])
        self.mpooling3d_2_1 = nn.MaxPool3d(kernel_size=[1,5,1],stride=[1,5,1])
        self.mpooling3d_1_2 = nn.MaxPool3d(kernel_size=[1,3,1],stride=[1,3,1])
        self.mpooling3d_2_2 = nn.MaxPool3d(kernel_size=[1,5,1],stride=[1,5,1])

    def forward(self, x):
        h = x   # (@,2,6,121,T_MAX)

        # Baseline: CNN+FC
        h_1 = h[:,0,:,:,:]                          # (@,2,6,121,T_MAX)=>(@,6,121,T_MAX)
        h_1 = torch.unsqueeze(h_1,dim=1)            # (@,6,121,T_MAX)=>(@,1,6,121,T_MAX)
        h_1 = relu(self.conv3d_1_1(h_1))            # (@,1,6,121,T_MAX)=>(@,16,6,117,T_MAX-4)
        h_1 = self.mpooling3d_1_1(h_1)              # (@,16,6,117,T_MAX-4)=>(@,16,6,39,T_MAX-4)
        h_1 = relu(self.conv3d_2_1(h_1))            # (@,16,6,39,T_MAX-4)=>(@,32,6,35,T_MAX-8)
        h_1 = self.mpooling3d_2_1(h_1)              # (@,32,6,35,T_MAX-8)=>(@,32,6,7,T_MAX-8)
        h_1 = relu(self.conv3d_3_1(h_1))            # (@,32,6,7,T_MAX-8)=>(@,32,6,5,T_MAX-10)
        h_1 = h_1.permute(0,2,4,1,3)                # (@,32,6,5,T_MAX-10)=>(@,6,T_MAX-10,32,5)
        h_1 = h_1.reshape((-1,6,self.T_MAX-10,32*5))# (@,6,T_MAX-10,32,5)=>(@,6,T_MAX-10,32*5)
        h_1 = self.dropout_1_1(h_1)
        h_1 = relu(self.fc_1_1(h_1))                  # (@,6,T_MAX-10,32*5)=>(@,6,T_MAX-10,128)
        h_1 = self.dropout_2_1(h_1)
        h_1 = relu(self.fc_2_1(h_1))                  # (@,6,T_MAX-10,128)=>(@,6,T_MAX-10,64)
        h_1 = relu(self.fc_3_1(h_1))                  # (@,6,T_MAX-10,64)=>(@,6,T_MAX-10,32)

        h_2 = h[:,1,:,:,:]                          # (@,2,6,121,T_MAX)=>(@,6,121,T_MAX)
        h_2 = torch.unsqueeze(h_2,dim=1)            # (@,6,121,T_MAX)=>(@,1,6,121,T_MAX)
        h_2 = relu(self.conv3d_1_2(h_2))            # (@,1,6,121,T_MAX)=>(@,16,6,117,T_MAX-4)
        h_2 = self.mpooling3d_1_2(h_2)              # (@,16,6,117,T_MAX-4)=>(@,16,6,39,T_MAX-4)
        h_2 = relu(self.conv3d_2_2(h_2))            # (@,16,6,39,T_MAX-4)=>(@,32,6,35,T_MAX-8)
        h_2 = self.mpooling3d_2_2(h_2)              # (@,32,6,35,T_MAX-8)=>(@,32,6,7,T_MAX-8)
        h_2 = relu(self.conv3d_3_2(h_2))            # (@,32,6,7,T_MAX-8)=>(@,32,6,5,T_MAX-10)
        h_2 = h_2.permute(0,2,4,1,3)                # (@,32,6,5,T_MAX-10)=>(@,6,T_MAX-10,32,5)
        h_2 = h_2.reshape((-1,6,self.T_MAX-10,32*5))# (@,6,T_MAX-10,32,5)=>(@,6,T_MAX-10,32*5)
        h_2 = self.dropout_1_2(h_2)
        h_2 = relu(self.fc_1_2(h_2))                  # (@,6,T_MAX-10,32*5)=>(@,6,T_MAX-10,128)
        h_2 = self.dropout_2_2(h_2)
        h_2 = relu(self.fc_2_2(h_2))                  # (@,6,T_MAX-10,128)=>(@,6,T_MAX-10,64)
        h_2 = relu(self.fc_3_2(h_2))                  # (@,6,T_MAX-10,64)=>(@,6,T_MAX-10,32)

        h = torch.cat((h_1,h_2),dim=3)              # (@,6,T_MAX-10,32)=>(@,6,T_MAX-10,64)
        h = h.reshape(-1,6*(self.T_MAX-10)*64)      # (@,6,T_MAX-10,64)=>(@,6*(T_MAX-10)*64)

        # Branch-1: Gesture class
        h = relu(self.fc_out_1(h))          # (@,6*(T_MAX-10)*64)=>(@,256)
        output = self.fc_out_2(h)        # (@,256)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        return output

class Baseline_RNN(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(Baseline_RNN, self).__init__()
        self.input_shape = input_shape  # [2,6,121,T_MAX]
        self.T_MAX = input_shape[3]
        self.class_num = class_num

        # CNN+FC
        self.fc_1 = nn.Linear(self.T_MAX*128, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.fc_3 = nn.Linear(512, 256)
        self.fc_4 = nn.Linear(256, 128)
        self.fc_5 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, self.class_num)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.2)
        self.dropout_4 = nn.Dropout(p=0.3)
        self.GRU = nn.GRU(input_size=6*121, hidden_size=128, num_layers=3, bias=True, batch_first=True, dropout = 0.2)

    def forward(self, x):
        h = x   # (@,2,6,121,T_MAX)

        # Baseline: CNN+GRU+Adv
        h = torch.linalg.norm(h,dim=1)          # (@,2,6,121,T_MAX)=>(@,6,121,T_MAX)

        h = h.permute(0,3,1,2)                  # (@,6,121,T_MAX)=>(@,T_MAX,6,121)
        h = h.reshape((-1,self.T_MAX,6*121))    # (@,T_MAX,6,121)=>(@,T_MAX,6*121)
        
        h,_ = self.GRU(h)                       # (@,T_MAX,6*121)=>(@,T_MAX,128)

        h = h.reshape(-1,self.T_MAX*128)        # (@,T_MAX,128)=>(@,T_MAX*128)

        # Branch-1: Gesture class
        h = relu(self.fc_1(h))                  # (@,T_MAX*128)=>(@,1024)
        h = self.dropout_1(h)
        h = relu(self.fc_2(h))                  # (@,1024)=>(@,512)
        h = self.dropout_2(h)
        h = relu(self.fc_3(h))                  # (@,512)=>(@,256)
        h = self.dropout_3(h)
        h = relu(self.fc_4(h))                  # (@,256)=>(@,128)
        h = self.dropout_4(h)
        h = relu(self.fc_5(h))                  # (@,128)=>(@,64)
        output = self.fc_out(h)                 # (@,64)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        return output

class Baseline_CMLP(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(Baseline_CMLP, self).__init__()
        self.input_shape = input_shape  # [2,6,121,T_MAX]
        self.T_MAX = input_shape[3]
        self.n_class = class_num

        # cMLP+rMLP
        self.complex_fc_1 = m_Linear(6*121*self.T_MAX, 2048)
        self.complex_fc_2 = m_Linear(2048, 1024)
        self.complex_fc_3 = m_Linear(1024, 512)
        self.complex_fc_4 = m_Linear(512, 256)
        self.complex_fc_5 = m_Linear(256, 128)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 64)
        self.fc_output = nn.Linear(64, self.n_class)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_6 = nn.Dropout(p=0.5)
        self.dropout_7 = nn.Dropout(p=0.5)

    def forward(self, x):
        h = x   # (@,2,6,121,T_MAX)

        # Baseline: cMLP+rMLP
        h = h.reshape((-1,2,6*121*self.T_MAX))    # (@,2,6,121,T_MAX)=>(@,2,6*121*T_MAX)

        h = relu(self.complex_fc_1(h))          # (@,2,6*121*T_MAX)=>(@,2,2048)
        h = self.dropout_2(h)
        h = relu(self.complex_fc_2(h))          # (@,2,2048)=>(@,2,1024)
        h = self.dropout_3(h)
        h = relu(self.complex_fc_3(h))          # (@,2,1024)=>(@,2,512)
        h = self.dropout_4(h)
        h = relu(self.complex_fc_4(h))          # (@,2,512)=>(@,2,256)
        h = self.dropout_5(h)
        h = relu(self.complex_fc_5(h))          # (@,2,256)=>(@,2,128)

        h = torch.linalg.norm(h,dim=1)          # (@,2,128)=>(@,128)

        h = relu(self.fc_1(h))                  # (@,128)=>(@,128)
        h = self.dropout_6(h)
        h = relu(self.fc_2(h))                  # (@,128)=>(@,128)
        h = self.dropout_7(h)
        h = relu(self.fc_3(h))                  # (@,128)=>(@,64)

        output = self.fc_output(h)              # (@,64)=>(@,n_class)

        return output

class Baseline_CCNN(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(Baseline_CCNN, self).__init__()
        self.input_shape = input_shape  # [2,6,121,T_MAX]
        self.T_MAX = input_shape[3]
        self.n_class = class_num

        # cCONV+cMLP+rMLP
        self.complex_fc_1 = m_Linear(32*7, 128)
        self.complex_fc_2 = m_Linear(128, 64)
        self.fc_1 = nn.Linear(6*(self.T_MAX-8)*64, 256)
        self.fc_out = nn.Linear(256, self.n_class)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.dropout_4 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.2)
        self.pconv3d_1 = m_pconv3d(in_channels=1,out_channels=16,kernel_size=[1,5,5],stride=[1,1,1],is_front_pconv_layer=True)
        self.pconv3d_2 = m_pconv3d(in_channels=16,out_channels=32,kernel_size=[1,5,5],stride=[1,1,1],is_front_pconv_layer=False)
        self.mpooling3d_1 = nn.MaxPool3d(kernel_size=[1,3,1],stride=[1,3,1])
        self.mpooling3d_2 = nn.MaxPool3d(kernel_size=[1,5,1],stride=[1,5,1])

    def forward(self, x):
        h = x   # (@,2,6,121,T_MAX)
        h = h.unsqueeze(dim=2)                      # (@,2,6,121,T_MAX)=>(@,2,R,6,121,T_MAX)

        # cCONV+cMLP+rMLP
        h = self.pconv3d_1(h)                       # (@,2,R,6,121,T_MAX)=>(@,2,16,6,117,T_MAX-4)
        h = h.reshape((-1,16,6,117,self.T_MAX-4))   # (@,2,16,6,117,T_MAX-4)=>(@*2,16,6,117,T_MAX-4)
        h = self.mpooling3d_1(h)                    # (@*2,16,6,117,T_MAX-4)=>(@*2,16,6,39,T_MAX-4)
        h = h.reshape((-1,2,16,6,39,self.T_MAX-4))  # (@*2,16,6,39,T_MAX-4)=>(@,2,16,6,39,T_MAX-4)

        h = self.pconv3d_2(h)                       # (@,2,16,6,39,T_MAX-4)=>(@,2,32,6,35,T_MAX-8)
        h = h.reshape((-1,32,6,35,self.T_MAX-8))    # (@,2,32,6,35,T_MAX-8)=>(@*2,32,6,35,T_MAX-8)
        h = self.mpooling3d_2(h)                    # (@*2,32,6,35,T_MAX-8)=>(@*2,32,6,7,T_MAX-8)
        h = h.reshape((-1,2,32,6,7,self.T_MAX-8))   # (@*2,32,6,7,T_MAX-8)=>(@,2,32,6,7,T_MAX-8)

        # cMLP
        h = h.permute(0,3,5,1,2,4)                  # (@,2,32,6,7,T_MAX-8)=>(@,6,T_MAX-8,2,32,7)
        h = h.reshape((-1,6,self.T_MAX-8,2,32*7))   # (@,6,T_MAX-8,2,32,7)=>(@,6,T_MAX-8,2,32*7)
        h = self.dropout_1(h)
        h = self.complex_fc_1(h)                    # (@,6,T_MAX-8,2,32*7)=>(@,6,T_MAX-8,2,128)
        h = self.dropout_2(h)
        h = self.complex_fc_2(h)                    # (@,6,T_MAX-8,2,128)=>(@,6,T_MAX-8,2,64)

        # rMLP
        h = torch.linalg.norm(h,dim=3)              # (@,6,T_MAX-8,2,64)=>(@,6,T_MAX-8,64)
        h = h.reshape((-1,6*(self.T_MAX-8)*64))     # (@,6,T_MAX-8,64)=>(@,6*(T_MAX-8)*64)
        h = self.dropout_3(h)
        h = relu(self.fc_1(h))                      # (@,6*(T_MAX-8)*64)=>(@,256)
        h = self.dropout_4(h)
        output = self.fc_out(h)                     # (@,256)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        return output

def train(model, n_epoch, optimizer, train_data_loader, valid_data_loader):
    # Template train function for classification problem
    train_loss = []
    valid_loss = []
    valid_acc = []
    for i_epoch in range(n_epoch):
        model.train()
        total_loss_this_epoch = 0
        for batch_idx, (data_batch, label_batch, domain_batch) in enumerate(train_data_loader):
            data_batch = data_batch.cuda()
            label_batch = label_batch.cuda()    # [@] (Note: NOT onehot)
            domain_batch = domain_batch.cuda()

            optimizer.zero_grad()
            if use_adversarial:
                label_pred_onehot, domain_pred_onehot = model(data_batch)
                loss = loss_function_adv(label_batch, domain_batch, label_pred_onehot, domain_pred_onehot)
            else:
                label_pred_onehot = model(data_batch)
                loss = loss_function(label_batch, label_pred_onehot)            
            loss.backward()
            optimizer.step()
            
            total_loss_this_epoch += loss.item()
            
            if batch_idx % 10 == 0:
                print('--------> Epoch: {}/{} loss: {:.4f} [{}/{} ({:.0f}%)]'.format(
                    i_epoch+1, n_epoch, loss.item() / len(data_batch), \
                        batch_idx * len(data_batch), len(train_data_loader.dataset), 100. * batch_idx / len(train_data_loader)), end='\r')
        model.eval()
        correct_count = total_count = 0
        total_valid_loss = 0
        for batch_idx, (data_batch, label_batch, domain_batch) in enumerate(valid_data_loader):
            if use_adversarial:
                label_pred_onehot, domain_pred_onehot = model(data_batch.cuda())    # [@,n_class]
                domain_pred_onehot = domain_pred_onehot.cpu()
            else:
                label_pred_onehot = model(data_batch.cuda())    # [@,n_class]
            label_pred_onehot = label_pred_onehot.cpu()
            label_pred = torch.argmax(label_pred_onehot, dim=-1)    # [@]
            correct_count += (label_pred == label_batch).sum().item()
            total_count += label_pred_onehot.shape[0]
            if use_adversarial:
                total_valid_loss += loss_function_adv(label_batch, domain_batch, label_pred_onehot, domain_pred_onehot).item()
            else:
                total_valid_loss += loss_function(label_batch, label_pred_onehot).item()                
        print('========> Epoch: {}/{} Loss: {:.4f}, valid accuracy: {:.1f}%({}/{})'.format(i_epoch+1, n_epoch, \
            total_loss_this_epoch / len(train_data_loader.dataset), 100*correct_count/total_count, correct_count, total_count) + ' '*20)
        train_loss.append(total_loss_this_epoch / len(train_data_loader.dataset))
        valid_loss.append(total_valid_loss      / len(valid_data_loader.dataset))
        valid_acc.append(100*correct_count / total_count)
    return train_loss, valid_loss, valid_acc

def test(model, data_loader):
    # Template test function for classification problem
    model.eval()
    with torch.no_grad():
        correct_count = 0
        total_count = 0
        for batch_idx, (data_batch, label_batch, domain_batch) in enumerate(data_loader):
            if use_adversarial:
                label_pred_onehot, domain_pred_onehot = model(data_batch.cuda())    # [@,n_class]
            else:
                label_pred_onehot = model(data_batch.cuda())    # [@,n_class]
            label_pred_onehot = label_pred_onehot.cpu()
            label_pred = torch.argmax(label_pred_onehot, dim=-1)    # [@]
            correct_count += (label_pred == label_batch).sum().item()
            total_count += label_pred_onehot.shape[0]
        print('Test accuracy: {:.1f}%({}/{})'.format(100*correct_count/total_count,correct_count,total_count))

# ======================== Start Here ========================
if len(sys.argv) < 2:
    print('Please specify which GPU to use ...')
    exit(0)
if (sys.argv[1] == '1' or sys.argv[1] == '0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    pass
else:
    print('Wrong GPU number, 0 or 1 supported!')
    exit(0)

# Load and reformat dataset
train_data_loader, valid_data_loader, test_data_loader, sample_shape = load_data_to_loader(data_dir=data_dir)

# Load or fabricate model
if use_existing_SLNet_model:
    print('Model loading...')
    model = torch.load(str_SLNet_name)
    print('Model loaded...')
else:
    print('Model building...')
    model = Baseline_EI(input_shape=sample_shape, class_num=N_PEOPLE)
    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    if str_optz == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=f_learning_rate)
    elif str_optz == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=f_learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=f_learning_rate)
    print('Model training...')
    train_loss, valid_loss, valid_acc = train(model=model, n_epoch=n_epoch, optimizer=optimizer, train_data_loader=train_data_loader, valid_data_loader=valid_data_loader)
    print('Model trained...')

# Test model
test(model=model, data_loader=test_data_loader)
