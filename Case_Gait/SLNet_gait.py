import os,sys,math
import scipy
import numpy as np
import scipy.io as scio

import torch, torchvision
import torch.nn as nn
from torch.nn.functional import relu, softmax, cross_entropy
from torch import sigmoid,tanh
from Modules.custom_layers_torch import m_cconv3d, m_Linear, m_Filtering, m_pconv3d
from SLNet_HiFi_Filter_trainer import SLNet_HiFi_Filter, complex_array_to_bichannel_float_tensor, bichannel_float_tensor_to_complex_array
from thop import profile


# Definition
use_existing_SLNet_model = False    # Note: NOT HiFiFilter model
W = 251                             # <====
wind_type = 'gaussian'              # <====
use_fused_spec_and_filter = True    # <====
use_singl_spec_and_filter = True    # <====
downsamp_ratio_for_stft = 10        # 100Hz / 10 = 10Hz        # <====
data_dir = 'xxx/' # <====
n_epoch = 100    # <====
# --------------------------
ID_LIST = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10', 'id11']
epoch_filter = 5000
W_all = [125,251,501]
W_all.sort()
str_SLNet_name = 'SLNet_gait_torch.pt'
str_HiFiFilter_name = 'xxx.pt'
mat_file_key = 'SPEC_W' + str(W)
T_MAX = 0
frac_for_valid = 0.1
frac_for_test = 0.05
str_optz = 'RMSprop'
f_learning_rate = 0.001
n_batch_size = 128


def normalize_data(data_1):
    # max=1 for each [F,T] spectrum
    # data(ndarray.complex)=>data_norm(ndarray.complex): [R,6,121,T]=>[R,6,121,T]
    data_1_abs = abs(data_1)
    data_1_max = data_1_abs.max(axis=(2,3),keepdims=True)     # [R,6,121,T]=>[R,6,1,1]
    data_1_max_rep = np.tile(data_1_max,(1,1,data_1_abs.shape[2],data_1_abs.shape[3]))    # [R,6,1,1]=>[R,6,121,T]
    data_1_norm = data_1 / data_1_max_rep   # [R,6,121,T]
    return  data_1_norm

def zero_padding(data, T_MAX):
    # data(list)=>data_pad(ndarray): ([R,6,121,t1/t2/...],...)=>[N,6,121,T_MAX]
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[3]
        tmp = np.pad(data[i], ((0,0),(0,0),(0,0),(T_MAX - t,0)), 'constant', constant_values = 0).tolist()
        data_pad.append(tmp)   # Front padding
    res = np.array(data_pad)
    return res

def complex_array_to_2_channel_float_array(data_complex):
    # data_complex(complex128/float64)=>data_float: [N,R,6,121,T_MAX]=>[N,2,R,6,121,T_MAX]
    data_complex = data_complex.astype('complex64')
    data_real = data_complex.real
    data_imag = data_complex.imag
    data_2_channel_float = np.stack((data_real, data_imag), axis=1)
    return data_2_channel_float

def fuse_spectrum(path_to_spec_W, HiFiFilter_list):
    # Use global: W/ W_all/ downsamp_ratio_for_stft/ str_HiFiFilter_name
    # path_to_spec_W: with resolution W (global var.)
    # Output: SPEC_FUSED: [R,C,F,T]~[3,6,121,T] R: Number of Resolution Candidates
    
    # Load all resolution SPEC, downsample, HiFiFilter
    spec_all_reso = []
    for _i in range(len(W_all)):
        w_i = W_all[_i]
        path_spec_Wi = path_to_spec_W.replace('W'+str(W), 'W'+str(w_i))
        if w_i in [31,61,999]:
            path_spec_Wi = path_spec_Wi.replace('100hz', '10hz')
            downsamp_ratio_for_stft = 1
        else:
            path_spec_Wi = path_spec_Wi.replace('10hz', '100hz')
            downsamp_ratio_for_stft = 10

        data_Wi = scio.loadmat(path_spec_Wi)['SPEC_W' + str(w_i)]  # [6,121,T]j

        # Downsample
        data_Wi = data_Wi[:,:,0::downsamp_ratio_for_stft]   # [6,121,T]j=>[6,121,t]j

        # Apply pre-trained HiFiFilter
        data_Wi = complex_array_to_bichannel_float_tensor(data_Wi)  # [6,121,t]j=>[6,121,2,t]
        data_Wi = data_Wi.permute(0,3,2,1)                          # [6,121,2,t]=>[6,t,2,121]
        HiFiFilter = HiFiFilter_list[_i]
        HiFiFilter.eval()
        with torch.no_grad():
            data_Wi = HiFiFilter(data_Wi.cuda()).cpu()              # [6,t,2,121]
        data_Wi = bichannel_float_tensor_to_complex_array(data_Wi)  # [6,T,121]j
        data_Wi = np.transpose(data_Wi,(0,2,1))     # [6,T,121]j=>[6,121,T]j

        # Stack
        spec_all_reso.append(np.abs(data_Wi))   # Abs anyway, because PCONV would abandon initial phase
    
    # SPEC fusion with TRS
    SPEC_FUSED = np.array(spec_all_reso)    # [R,C,F,T]~[3,6,121,T]

    return SPEC_FUSED

def filter_spectrum(file_path, HiFiFilter):
    # Use global: mat_file_key/ downsamp_ratio_for_stft
    data_W = scio.loadmat(file_path)[mat_file_key]  # [6,121,T]j

    # Downsample
    data_W = data_W[:,:,0::downsamp_ratio_for_stft]   # [6,121,T]j=>[6,121,t]j

    # Apply pre-trained HiFiFilter
    data_W = complex_array_to_bichannel_float_tensor(data_W)  # [6,121,t]j=>[6,121,2,t]
    data_W = data_W.permute(0,3,2,1)                          # [6,121,2,t]=>[6,t,2,121]
    HiFiFilter.eval()
    with torch.no_grad():
        data_W = HiFiFilter(data_W.cuda()).cpu()              # [6,t,2,121]
    data_W = bichannel_float_tensor_to_complex_array(data_W)  # [6,t,121]j
    data_W = np.transpose(data_W,(0,2,1))     # [6,t,121]j=>[6,121,t]j
    
    ret = data_W
    return ret

def load_HiFiFilter_list():
    # Use global: W/ W_all/ str_HiFiFilter_name
    HFF_list = []
    for w_i in W_all:
        path_HFF_Wi = str_HiFiFilter_name.replace('W'+str(W), 'W'+str(w_i))
        HFF_Wi = torch.load(path_HFF_Wi)
        HFF_list.append(HFF_Wi.cuda())
    return HFF_list

def load_data_to_array(path_to_data):
    # Need customization
    # data: [N,2,6,121,T_MAX]
    # label: [N,]
    global T_MAX
    print('Loading data from ' + str(path_to_data))

    # Load HiFiFilter for fuse case or single reso case
    if use_fused_spec_and_filter:
        print('Using multiple reso. spectrum with respective HiFiFilter...')
        print('Fusing ' + str(W_all))
        HiFiFilter_list = load_HiFiFilter_list()
    elif use_singl_spec_and_filter:
        HiFiFilter = torch.load(str_HiFiFilter_name).cuda()
        print('Using single reso. spectrum with HiFiFilter...' + str_HiFiFilter_name)
    else:
        print('Using single reso. spectrum without HiFiFilter...')

    # Load data
    data = []
    label = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:

            file_path = os.path.join(data_root,data_file_name)
            try:
                # Data Selection
                label_1_name = data_file_name.split('-')[0]
                if label_1_name in ID_LIST:
                    label_1 = ID_LIST.index(label_1_name)+1
                else:
                    continue

                # Fuse spec or filter single reso spec or original single reso spec
                if use_fused_spec_and_filter:
                    data_1 = fuse_spectrum(file_path, HiFiFilter_list)  # [R,6,121,t]j
                    if data_1 == []:
                        continue
                elif use_singl_spec_and_filter:
                    data_1 = filter_spectrum(file_path, HiFiFilter) # [6,121,t]j
                    if data_1 == []:
                        continue
                    data_1 = np.expand_dims(data_1, axis=0)         # [6,121,t]j=>[1,6,121,t]j
                else:
                    data_1 = scio.loadmat(file_path)[mat_file_key]  # [6,121,T]j
                    # Downsample
                    data_1 = data_1[:,:,0::downsamp_ratio_for_stft] # [6,121,T]j=>[6,121,t]j
                    data_1 = np.expand_dims(data_1, axis=0)         # [6,121,t]j=>[1,6,121,t]j

                # Skip rx is not equal to 6
                if data_1.shape[1] is not 6:
                    print('Skipping ' + str(data_file_name) + ', Rx not 6')
                    continue

                # Skip nan
                if np.sum(np.isnan(data_1)):
                    print('Skipping ' + str(data_file_name))
                    continue

                # Normalization
                data_normed_1 = normalize_data(data_1)  # [R,C,F,T] ~ [R,6,121,t]

                # Update T_MAX
                if T_MAX < np.array(data_1).shape[3]:
                    T_MAX = np.array(data_1).shape[3]                
            except Exception as e:
                print(str(e))
                continue

            # Save List
            data.append(data_normed_1.tolist())
            label.append(label_1-1)

    # Zero-padding
    data = zero_padding(data, T_MAX)    # ([R,C,F,t],...)=>[N,R,C,F,T]

    # Convert from complex128 to 2_channel_float32
    data = complex_array_to_2_channel_float_array(data)    # [N,R,C,F,T]~[N,R,6,121,T]=>[N,2,R,6,121,T]

    # Convert label to ndarray
    label = np.array(label)

    # data(ndarray): [N,2,R,6,121,T_MAX], label(ndarray): [N,]
    return data, label

def load_data_to_loader(data_dir):
    # Template function
    print('Loading data...')
    # Load numpy => data,label (numpy array)
    data, label = load_data_to_array(data_dir)
    n_class = np.max(label) + 1

    # Numpy => torch.utils.data.TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.tensor(data,dtype=torch.float32), torch.tensor(label,dtype=torch.int64)) # Note: float32
    sample_shape = np.array(dataset[0][0].shape)
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

    print('Loaded {} samples with {} classes, split {}/{}/{} for train/valid/test, data shape: {}, data dtype: {}'.format(\
        sample_count,n_class,train_sample_count,valid_sample_count,test_sample_count,sample_shape,sample_dtype))
    return train_data_loader, valid_data_loader, test_data_loader, sample_shape, n_class

def loss_function(label_batch, label_pred):
    # Need customization
    # label_batch:[@], label_pred:[@,n_class]
    return nn.CrossEntropyLoss()(label_pred, label_batch)   # LogSoftmax + NLLLoss

class SLNet(nn.Module):
    # Need customization
    def __init__(self, input_shape, n_class):
        super(SLNet, self).__init__()
        self.input_shape = input_shape  # [2,R,6,121,T_MAX]
        self.n_class = n_class
        self.IN_CHANNEL = input_shape[1]
        self.T_MAX = input_shape[4]

        # pconv+FC
        self.complex_fc_1 = m_Linear(32*7, 128)
        self.complex_fc_2 = m_Linear(128, 64)
        self.fc_1 = nn.Linear(32*7, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 32)
        self.fc_4 = nn.Linear(6*(self.T_MAX-8)*32, 256)
        self.fc_5 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, self.n_class)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.dropout_4 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.2)
        self.pconv3d_1 = m_pconv3d(in_channels=self.IN_CHANNEL,out_channels=16,kernel_size=[1,5,5],stride=[1,1,1],is_front_pconv_layer=True)
        self.pconv3d_2 = m_pconv3d(in_channels=16,out_channels=32,kernel_size=[1,5,5],stride=[1,1,1],is_front_pconv_layer=False)
        self.mpooling3d_1 = nn.MaxPool3d(kernel_size=[1,3,1],stride=[1,3,1])
        self.mpooling3d_2 = nn.MaxPool3d(kernel_size=[1,5,1],stride=[1,5,1])

    def forward(self, x):
        h = x   # [@,2,R,C,F,T] ~ (@,2,1,6,121,T_MAX) or (@,2,3,6,121,T_MAX)

        # pconv
        h = self.pconv3d_1(h)                       # (@,2,R,6,121,T_MAX)=>(@,2,16,6,117,T_MAX-4)
        h = h.reshape((-1,16,6,117,self.T_MAX-4))   # (@,2,16,6,117,T_MAX-4)=>(@*2,16,6,117,T_MAX-4)
        h = self.mpooling3d_1(h)                    # (@*2,16,6,117,T_MAX-4)=>(@*2,16,6,39,T_MAX-4)
        h = h.reshape((-1,2,16,6,39,self.T_MAX-4))  # (@*2,16,6,39,T_MAX-4)=>(@,2,16,6,39,T_MAX-4)

        h = self.pconv3d_2(h)                       # (@,2,16,6,39,T_MAX-4)=>(@,2,32,6,35,T_MAX-8)
        h = h.reshape((-1,32,6,35,self.T_MAX-8))    # (@,2,32,6,35,T_MAX-8)=>(@*2,32,6,35,T_MAX-8)
        h = self.mpooling3d_2(h)                    # (@*2,32,6,35,T_MAX-8)=>(@*2,32,6,7,T_MAX-8)
        h = h.reshape((-1,2,32,6,7,self.T_MAX-8))   # (@*2,32,6,7,T_MAX-8)=>(@,2,32,6,7,T_MAX-8)

        # Complex FC
        h = h.permute(0,3,5,1,2,4)                  # (@,2,32,6,7,T_MAX-8)=>(@,6,T_MAX-8,2,32,7)
        h = h.reshape((-1,6,self.T_MAX-8,2,32*7))   # (@,6,T_MAX-8,2,32,7)=>(@,6,T_MAX-8,2,32*7)
        h = self.dropout_1(h)
        h = self.complex_fc_1(h)                    # (@,6,T_MAX-8,2,32*7)=>(@,6,T_MAX-8,2,128)
        h = self.dropout_2(h)
        h = self.complex_fc_2(h)                    # (@,6,T_MAX-8,2,128)=>(@,6,T_MAX-8,2,64)

        # FC
        h = torch.linalg.norm(h,dim=3)              # (@,6,T_MAX-8,2,64)=>(@,6,T_MAX-8,64)
        h = relu(self.fc_3(h))                      # (@,6,T_MAX-8,64)=>(@,6,T_MAX-8,32)
        h = h.reshape((-1,6*(self.T_MAX-8)*32))     # (@,6,T_MAX-8,32)=>(@,6*(T_MAX-8)*32)
        h = self.dropout_3(h)
        h = relu(self.fc_4(h))                      # (@,6*(T_MAX-8)*32)=>(@,256)
        h = self.dropout_4(h)
        h = relu(self.fc_5(h))                      # (@,256)=>(@,128)
        h = self.dropout_5(h)
        output = self.fc_out(h)           # (@,128)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        return output

def train(model, n_epoch, optimizer, train_data_loader, valid_data_loader):
    # Template train function for classification problem
    train_loss = []
    valid_loss = []
    valid_acc = []
    for i_epoch in range(n_epoch):
        model.train()
        total_loss_this_epoch = 0
        for batch_idx, (data_batch, label_batch) in enumerate(train_data_loader):
            data_batch = data_batch.cuda()
            label_batch = label_batch.cuda()    # [@] (Note: NOT onehot)

            optimizer.zero_grad()
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
        for batch_idx, (data_batch, label_batch) in enumerate(valid_data_loader):
            label_pred_onehot = model(data_batch.cuda()).cpu()      # [@,n_class]
            label_pred = torch.argmax(label_pred_onehot, dim=-1)    # [@]
            correct_count += (label_pred == label_batch).sum().item()
            total_count += label_pred_onehot.shape[0]
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
        for batch_idx, (data_batch,label_batch) in enumerate(data_loader):
            label_pred_onehot = model(data_batch.cuda()).cpu()      # [@,n_class]
            label_pred = torch.argmax(label_pred_onehot, dim=-1)    # [@]
            correct_count += (label_pred == label_batch).sum().item()
            total_count += label_pred_onehot.shape[0]
        print('Test accuracy: {:.1f}%({}/{})'.format(100*correct_count/total_count,correct_count,total_count))

# ======================== Start Here ========================
if len(sys.argv) < 1:
    print('Please specify which GPU to use ...')
    exit(0)
if (sys.argv[1] == '1' or sys.argv[1] == '0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    pass
else:
    print('Wrong GPU number, 0 or 1 supported!')
    exit(0)

# Load and reformat dataset
train_data_loader, valid_data_loader, test_data_loader, sample_shape, n_class = load_data_to_loader(data_dir=data_dir)

# Load or fabricate model
if use_existing_SLNet_model:
    print('Model loading...')
    model = torch.load(str_SLNet_name)
    print('Model loaded...')
else:
    print('Model building...')
    model = SLNet(input_shape=sample_shape, n_class=n_class)
    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    if str_optz == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=f_learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=f_learning_rate)
    print('Model training...')
    train_loss, valid_loss, valid_acc = train(model=model, n_epoch=n_epoch, optimizer=optimizer, train_data_loader=train_data_loader, valid_data_loader=valid_data_loader)
    print('Model trained...')

# Test model
test(model=model, data_loader=test_data_loader)
