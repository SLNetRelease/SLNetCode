import os,sys,math
import scipy
import numpy as np
import scipy.io as scio
from scipy import interpolate

import torch, torchvision
import torch.nn as nn
from torch.nn.functional import relu, softmax, cross_entropy
from torch import sigmoid,tanh
from thop import profile
from sklearn.metrics import confusion_matrix


# Definition
str_baseline_model = 'densenet121'      # <==== alexnet, vgg11/16, resnet18/34/50/101, densenet121
downsamp_ratio_for_stft = 40        # 1000Hz / 100 = 10Hz
data_dir = 'xxx/'
mat_file_key = 'DFS'
# --------------------------
T_MAX = 0
frac_for_valid = 0.1
frac_for_test = 0.1
str_optz = 'RMSprop'
f_learning_rate = 0.001
n_batch_size = 128
n_epoch = 60


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

def load_data_to_array(path_to_data):
    # Need customization
    # data: [N,6,121,T_MAX]
    # label: [N,]
    global T_MAX
    print('Loading data from ' + str(path_to_data))

    # Load data
    data = []
    label = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:

            file_path = os.path.join(data_root,data_file_name)
            try:
                # Data Selection
                label_1_name = data_file_name.split('_')[0]
                if label_1_name == 'fall':
                    label_1 = 1
                elif label_1_name == 'nonfall':
                    label_1 = 0
                else:
                    print('Wrong fall/nonfall prefix, skiping...')
                    continue

                data_1 = np.abs(scio.loadmat(file_path)[mat_file_key])  # [T,61]
                # Interpolation
                f = scipy.interpolate.interp2d(np.arange(0,data_1.shape[1]), np.arange(0,data_1.shape[0]), data_1)
                data_1 = f(np.linspace(0,data_1.shape[1],121), np.arange(0,data_1.shape[0]))    # [T,61] => [T,121]
                data_1 = np.expand_dims(data_1.transpose(), axis=0) # [T,121]=>[1,121,T]
                # Downsample
                data_1 = data_1[:,:,0::downsamp_ratio_for_stft] # [1,121,T]=>[1,121,t]

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
            label.append(label_1)

    # Zero-padding
    data = zero_padding(data, T_MAX)    # ([6,121,t1],...)=>[N,6,121,T]

    # Convert label to ndarray
    label = np.array(label)

    # data(ndarray): [N,6,121,T_MAX], label(ndarray): [N,]
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

    print('Loaded {} samples, split {}/{}/{} for train/valid/test, data shape: {}, data dtype: {}'.format(\
        sample_count,train_sample_count,valid_sample_count,test_sample_count,sample_shape,sample_dtype))
    return train_data_loader, valid_data_loader, test_data_loader, sample_shape, n_class

class Alexnet(nn.Module):
    # Need customization
    def __init__(self):
        super(Alexnet, self).__init__()
        self.n_class = 11

        # Alexnet
        self.fc_1 = nn.Linear(256*6*6, 256*6*6)
        self.fc_2 = nn.Linear(256*6*6, 256*6*6)
        self.fc_out = nn.Linear(256*6*6, self.n_class)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.conv2d_1 = nn.Conv2d(1,    96, kernel_size=(13,4), stride=(2,1), padding=(0,0))
        self.conv2d_2 = nn.Conv2d(96,  256, kernel_size=(5,5),   stride=(1,1), padding=(2,2))
        self.conv2d_3 = nn.Conv2d(256, 384, kernel_size=(3,3),   stride=(1,1), padding=(1,1))
        self.conv2d_4 = nn.Conv2d(384, 384, kernel_size=(3,3),   stride=(1,1), padding=(1,1))
        self.conv2d_5 = nn.Conv2d(384, 256, kernel_size=(3,3),   stride=(1,1), padding=(1,1))
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=(2,1))
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        h = x   # (@,1,121,T_MAX=37)

        h = self.conv2d_1(h)        # (@,1,121,30)=>(@,96,55,27)
        h = relu(h)
        h = self.maxpool2d_1(h)     # (@,96,55,27)=>(@,96,27,27)
        
        h = self.conv2d_2(h)        # (@,96,27,27)=>(@,256,27,27)
        h = relu(h)
        h = self.maxpool2d_2(h)     # (@,256,27,27)=>(@,256,13,13)

        h = self.conv2d_3(h)        # (@,256,13,13)=>(@,384,13,13)
        h = relu(h)
        h = self.conv2d_4(h)        # (@,384,13,13)=>(@,384,13,13)
        h = relu(h)
        h = self.conv2d_5(h)        # (@,384,13,13)=>(@,256,13,13)
        h = relu(h)

        h = self.maxpool2d_3(h)     # (@,256,13,13)=>(@,256,6,6)
        
        h = h.reshape(-1, 256*6*6)  # (@,256,6,6)=>(@,256*6*6)
        h = self.dropout_1(h)
        h = self.fc_1(h)            # (@,256*6*6)=>(@,256*6*6)
        h = relu(h)
        h = self.dropout_2(h)
        h = self.fc_2(h)            # (@,256*6*6)=>(@,256*6*6)
        h = relu(h)

        output = self.fc_out(h)     # (@,256*6*6)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        return output

def loss_function(label_batch, label_pred):
    # Need customization
    # label_batch:[@], label_pred:[@,n_class]
    return nn.CrossEntropyLoss()(label_pred, label_batch)   # LogSoftmax + NLLLoss

class BaselineWrapperModel(nn.Module):
    def __init__(self, input_shape, baseline_model, model_type, n_class):
        super(BaselineWrapperModel, self).__init__()
        self.input_shape = input_shape  # [6,121,T_MAX]
        self.baseline_model = baseline_model
        self.input_conv2d = nn.Conv2d(in_channels=1, out_channels =3, kernel_size=(1,1))
        self.model_type = model_type
        self.n_class = 2
        
        # Finetuning baseline model
        for param in self.baseline_model.parameters():
            param.requires_grad = True
        
        # Adapt output layer
        if 'alexnet' in self.model_type:
            pass
        elif 'vgg' in self.model_type:
            self.rear_layer_inputs = self.baseline_model.classifier[-1].in_features
            self.baseline_model.classifier[-1] = nn.Linear(self.rear_layer_inputs, self.n_class)
        elif 'resnet' in self.model_type:
            self.rear_layer_inputs = self.baseline_model.fc.in_features
            self.baseline_model.fc = nn.Linear(self.rear_layer_inputs, self.n_class)
        elif 'densenet' in self.model_type:
            self.rear_layer_inputs = self.baseline_model.classifier.in_features
            self.baseline_model.classifier = nn.Linear(self.rear_layer_inputs, self.n_class)
        
    def forward(self, x):
        h = x   # [@,C,F,T] ~ [@,6,121,T_MAX]

        if 'alexnet' not in self.model_type:
            h = self.input_conv2d(h)    # [@,6,121,T_MAX]=>[@,3,121,T_MAX]
        output = self.baseline_model(h)
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
            # Confusion Matrix
            cm = confusion_matrix(label_batch, label_pred)
            print(cm)
            cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=2)
            print(cm)
        print('Test accuracy: {:.1f}%({}/{})'.format(100*correct_count/total_count,correct_count,total_count))

# ======================== Start Here ========================
if len(sys.argv) < 2:
    print('Please specify which GPU to use ...')
    exit(0)
if (sys.argv[1] == '1' or sys.argv[1] == '0'):
    # os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    pass
else:
    print('Wrong GPU number, 0 or 1 supported!')
    exit(0)

# Load and reformat dataset
train_data_loader, valid_data_loader, test_data_loader, sample_shape, n_class = load_data_to_loader(data_dir=data_dir)

# Load and wrap model
if str_baseline_model == 'alexnet':
    baseline_model = Alexnet()
elif str_baseline_model == 'vgg11':
    baseline_model = torchvision.models.vgg11(pretrained=False)
elif str_baseline_model == 'vgg16':
    baseline_model = torchvision.models.vgg16(pretrained=True)
elif str_baseline_model == 'resnet18':
    baseline_model = torchvision.models.resnet18(pretrained=False)
elif str_baseline_model == 'resnet34':
    baseline_model = torchvision.models.resnet34(pretrained=True)
elif str_baseline_model == 'resnet50':
    baseline_model = torchvision.models.resnet50(pretrained=True)
elif str_baseline_model == 'resnet101':
    baseline_model = torchvision.models.resnet101(pretrained=True)
elif str_baseline_model == 'densenet121':
    baseline_model = torchvision.models.densenet121(pretrained=True)
else:
    print('Not supported baseline model...')
    exit(0)
print('Model building from pre-trained ' + str_baseline_model + '...')
model = BaselineWrapperModel(input_shape=sample_shape, baseline_model=baseline_model, model_type=str_baseline_model, n_class=n_class)
model = model.cuda()
model = nn.DataParallel(model).cuda()

# Train model
print('Model training...')
if str_optz == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=f_learning_rate)
else:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=f_learning_rate)
train_loss, valid_loss, valid_acc = train(model=model, n_epoch=n_epoch, optimizer=optimizer, train_data_loader=train_data_loader, valid_data_loader=valid_data_loader)
print('Model trained...')

# Test model
test(model=model, data_loader=test_data_loader)
