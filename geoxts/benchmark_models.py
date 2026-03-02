
__all__ = ['build_model', 'count_parameters', 'random_seed', 'XCM_LSTM', 'LSTM_XCM','LSTM_FCN_2dCNN','XCM_local','LSTM_FCN_Base','LSTM_2dCNN', ]


import torch
import random
import numpy as np
import torch.nn as nn
from tsai.imports import default_device
from tsai.models.layers import Conv2d, Conv1d, BatchNorm, Squeeze, Unsqueeze, Concat, GAP1d, GACP1d, Reshape, Permute, ConvBlock, SqueezeExciteBlock, Module
from fastai.layers import *


def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

def pv(text, verbose):
    if verbose: print(text)

def build_model(arch, c_in=None, c_out=None, seq_len=None, dls=None, device=None,d=None, verbose=False, pretrained=False, weights_path=None, exclude_head=True, init=None, arch_config={}, **kwargs):

    if device is None: device = default_device()
    if dls is not None:
        if c_in is None : c_in = dls.vars
        if c_out is None: c_out = dls.c
        if seq_len is None: seq_len= dls.len
        if d is None :d= dls.d

    try:
        
        model = arch(c_in, c_out, seq_len=seq_len, **arch_config, **kwargs).to(device=device)
        pv(f'arch: {arch.__name__}(c_in={c_in} c_out={c_out} seq_len={seq_len} arch_config={arch_config} kwargs={kwargs})', verbose)
    except:
        try:
            model = (arch(c_in=c_in, c_out=c_out, **arch_config, **kwargs)).to(device=device)
            pv(f'arch: {arch.__name__}(c_in={c_in} c_out={c_out} arch_config={arch_config} kwargs={kwargs})', verbose)
        except:
            print('Please check the model parameters and the build_model function')

    return model 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def random_seed(seed_value, use_cuda):
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False




#XCM model implementation
class XCM_local(Module):
    def __init__(self, c_in:int, c_out:int, seq_len:int=None, nf:int=128, window_perc:float=1., flatten:bool=False, custom_head:callable=None, 
                 concat_pool:bool=False, fc_dropout:float=0., bn:bool=False, y_range:tuple=None, **kwargs):
        
        #super().__init__()

        window_size = int(round(seq_len * window_perc, 0))
        self.conv2dblock = nn.Sequential(*[Unsqueeze(1), Conv2d(1, nf, kernel_size=(1, window_size), padding='same'), BatchNorm(nf), nn.ReLU()])
        self.conv2d1x1block = nn.Sequential(*[nn.Conv2d(nf, 1, kernel_size=1), nn.ReLU(), Squeeze(1)])
        self.conv1dblock = nn.Sequential(*[Conv1d(c_in, nf, kernel_size=window_size, padding='same'), BatchNorm(nf, ndim=1), nn.ReLU()])
        self.conv1d1x1block = nn.Sequential(*[nn.Conv1d(nf, 1, kernel_size=1), nn.ReLU()])
        self.concat = Concat()
        self.conv1d = nn.Sequential(*[Conv1d(c_in + 1, nf, kernel_size=window_size, padding='same'), BatchNorm(nf, ndim=1), nn.ReLU()])
            
        self.head_nf = nf
        self.c_out = c_out
        self.seq_len = seq_len
        if custom_head: self.head = custom_head(self.head_nf, c_out, seq_len, **kwargs)
        else: self.head = self.create_head(self.head_nf, c_out, seq_len, flatten=flatten, concat_pool=concat_pool, 
                                           fc_dropout=fc_dropout, bn=bn, y_range=y_range)

            
    def forward(self, x):
        x1 = self.conv2dblock(x)
        x1 = self.conv2d1x1block(x1)
        x2 = self.conv1dblock(x)
        x2 = self.conv1d1x1block(x2)
        out = self.concat((x2, x1))
        out = self.conv1d(out)
        out = self.head(out)
        return out
    

    def create_head(self, nf, c_out, seq_len=None, flatten=False, concat_pool=False, fc_dropout=0., bn=False, y_range=None):
        if flatten: 
            nf *= seq_len
            layers = [Reshape()]
        else: 
            if concat_pool: nf *= 2
            layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)
    


class LSTM_XCM(Module):

    def __init__(self, c_in:int, c_out:int, seq_len:int=None, hidden_size=100, shuffle=True, cell_dropout=0, bias=True, rnn_dropout=0.8, bidirectional=False, 
                 nf:int=128, window_perc:float=1., flatten:bool=False, custom_head:callable=None, 
                 concat_pool:bool=False, fc_dropout:float=0., bn:bool=False, y_range:tuple=None, **kwargs):

        #super().__init__()

        if shuffle: assert seq_len is not None, 'need seq_len if shuffle=True'
        # RNN - first arg is usually c_in. Authors modified this to seq_len by not permuting x. This is what they call shuffled data.
        self.rnn = nn.LSTM(seq_len if shuffle else c_in, hidden_size, num_layers=1, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        window_size = int(round(seq_len * window_perc, 0))
        self.conv2dblock = nn.Sequential(*[Unsqueeze(1), Conv2d(1, nf, kernel_size=(1, window_size), padding='same'), BatchNorm(nf), nn.ReLU()])
        self.conv2d1x1block = nn.Sequential(*[nn.Conv2d(nf, 1, kernel_size=1), nn.ReLU(), Squeeze(1)])
        self.conv1dblock = nn.Sequential(*[Conv1d(c_in, nf, kernel_size=window_size, padding='same'), BatchNorm(nf, ndim=1), nn.ReLU()])
        self.conv1d1x1block = nn.Sequential(*[nn.Conv1d(nf, 1, kernel_size=1), nn.ReLU()])
        #self.conv1d = nn.Sequential(*[Conv1d(c_in + 1, nf, kernel_size=window_size, padding='same'), BatchNorm(nf, ndim=1), nn.ReLU()])   
        self.head_nf = nf
        self.c_out = c_out
        self.seq_len = seq_len
        
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = nn.Linear(hidden_size * (1 + bidirectional) + (c_in+1)*seq_len, c_out)
        

            
    def forward(self, x):
        #XCM
        #2D CNN
        x1 = self.conv2dblock(x)
        x1 = self.conv2d1x1block(x1)
        #print(x1.reshape((x1.shape[0], x.shape[1]*x.shape[2])).shape)
        #1dCNN
        x2 = self.conv1dblock(x)
        x2 = self.conv1d1x1block(x2)
        #print(x2.squeeze().shape)
        # RNN
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        x3= self.rnn_dropout(last_out)
        #x3 = torch.unsqueeze(x3,1)
        #print(torch.unsqueeze(x3,1).shape)
        x2s = x2.squeeze(dim=1)
        x1 = x1.reshape((x1.shape[0], x.shape[1]*x.shape[2]))
        if x2s.ndim == 2: 
            out = self.concat([x3, x2s, x1])
        #else: out = self.concat([x3, x2, x1.reshape((x1.shape[0], x.shape[1]*x.shape[2]))])
        #print(x6.shape)
        else:
            print(x3.ndim, x2.ndim, x2s.ndim, x1.ndim)
            print(x2.shape)
            out = self.concat([x3, x2s, x1])
            
        out = self.fc_dropout(out)
        out = self.fc(out)
        
        return out
    
    def create_head(self, nf, c_out, seq_len=None, flatten=False, concat_pool=False, fc_dropout=0.1, bn=False, y_range=None):
        if flatten: 
            nf *= seq_len
            layers = [Reshape()]
        else: 
            if concat_pool: nf *= 2
            layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)



class XCM_LSTM(Module):

    def __init__(self, c_in:int, c_out:int, seq_len:int=None, hidden_size=201, shuffle=True, cell_dropout=0, bias=True, rnn_dropout=0.8, bidirectional=False, 
                 nf:int=128, window_perc:float=1., flatten:bool=False, custom_head:callable=None, 
                 concat_pool:bool=False, fc_dropout:float=0.1, bn:bool=False, y_range:tuple=None, **kwargs):
        
        """
        - c_in:
        - c_out: 
        
        """
        #super().__init__()

        if shuffle: assert seq_len is not None, 'need seq_len if shuffle=True'
        # RNN - first arg is usually c_in. Authors modified this to seq_len by not permuting x. This is what they call shuffled data.
        self.rnn = nn.LSTM(seq_len if shuffle else c_in, hidden_size, num_layers=1, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        window_size = int(round(seq_len * window_perc, 0))
        self.conv2dblock = nn.Sequential(*[Unsqueeze(1), Conv2d(1, nf, kernel_size=(1, window_size), padding='same'), BatchNorm(nf), nn.ReLU()])
        self.conv2d1x1block = nn.Sequential(*[nn.Conv2d(nf, 1, kernel_size=1), nn.ReLU(), Squeeze(1)])
        self.conv1dblock = nn.Sequential(*[Conv1d(c_in, nf, kernel_size=window_size, padding='same'), BatchNorm(nf, ndim=1), nn.ReLU()])
        self.conv1d1x1block = nn.Sequential(*[nn.Conv1d(nf, 1, kernel_size=1), nn.ReLU()])
        
        self.conv1d = nn.Sequential(*[Conv1d(c_in + 1, nf, kernel_size=window_size, padding='same'), BatchNorm(nf, ndim=1), nn.ReLU()])
            
        self.head_nf = nf
        self.c_out = c_out
        self.seq_len = seq_len
        if custom_head: self.head = custom_head(self.head_nf, c_out, seq_len, **kwargs)
        else: self.head = self.create_head(self.head_nf, c_out, seq_len, flatten=flatten, concat_pool=concat_pool, 
                                           fc_dropout=fc_dropout, bn=bn, y_range=y_range)
        self.gap = GAP1d(1)
        # Common
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = nn.Linear(hidden_size * (1 + bidirectional) + nf, c_out)
            
    def forward(self, x):
        #XCM
        #2D CNN
        x1 = self.conv2dblock(x)
        x1 = self.conv2d1x1block(x1)
        #print(x1.shape)
        #1dCNN
        x2 = self.conv1dblock(x)
        x2 = self.conv1d1x1block(x2)
        #print(x2.shape)
        x3 = self.concat((x2, x1))
        #x3 = self.head(x3)
        # RNN
        rnn_input = self.shuffle(x3) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        x4= self.rnn_dropout(last_out)
        #x4 = torch.unsqueeze(x4,1)
        
        x5 = self.conv1d(x3)
        x5 = self.gap(x5)
        
        x6 = self.concat((x4,x5))
        #print(x6.shape)
        x6 = self.fc_dropout(x6)
        out = self.fc(x6)
        #print(x5.shape())
        return out
    
    def create_head(self, nf, c_out, seq_len=None, flatten=False, concat_pool=False, fc_dropout=0.1, bn=False, y_range=None):
        if flatten: 
            nf *= seq_len
            layers = [Reshape()]
        else: 
            if concat_pool: nf *= 2
            layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)
    

    
class LSTM_FCN_Base(Module):

    def __init__(self, c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout=0., conv_layers=[128, 256, 128], kss=[7, 5, 3], se=0):
        
        #super().__init__()

        if shuffle: assert seq_len is not None, 'need seq_len if shuffle=True'
            
        # RNN
        self.rnn = nn.LSTM(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        
        # FCN
        assert len(conv_layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, conv_layers[0], kss[0])
        self.se1 = SqueezeExciteBlock(conv_layers[0], se) if se != 0 else noop
        self.convblock2 = ConvBlock(conv_layers[0], conv_layers[1], kss[1])
        self.se2 = SqueezeExciteBlock(conv_layers[1], se) if se != 0 else noop
        self.convblock3 = ConvBlock(conv_layers[1], conv_layers[2], kss[2])
        self.gap = GAP1d(1)
        
        # Common
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = nn.Linear(hidden_size * (1 + bidirectional) + conv_layers[-1], c_out)
        

    def forward(self, x):  
        # RNN
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)
        
        # FCN
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x)
       
        # Concat
        x = self.concat([last_out, x])
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x   
    


class LSTM_FCN_2dCNN(Module):

    def __init__(self, c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout=0., conv_layers=[128, 256, 128], kss=[7, 5, 3], se=0):
        
        #super().__init__()

        if shuffle: assert seq_len is not None, 'need seq_len if shuffle=True'
            
        # RNN
        self.rnn = nn.LSTM(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        # FCN
        assert len(conv_layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, conv_layers[0], kss[0])
        self.se1 = SqueezeExciteBlock(conv_layers[0], se) if se != 0 else noop
        self.convblock2 = ConvBlock(conv_layers[0], conv_layers[1], kss[1])
        self.se2 = SqueezeExciteBlock(conv_layers[1], se) if se != 0 else noop
        self.convblock3 = ConvBlock(conv_layers[1], conv_layers[2], kss[2])
        self.gap = GAP1d(1)

        #2dCNN
        self.conv2dblock = nn.Sequential(*[Unsqueeze(1), Conv2d(1, conv_layers[0], kernel_size=(1, seq_len), padding='same'), BatchNorm(conv_layers[0]), nn.ReLU()])
        self.conv2d1x1block = nn.Sequential(*[nn.Conv2d(conv_layers[0], 1, kernel_size=1), nn.ReLU(), Squeeze(1)])
        self.conv1dblock = nn.Sequential(*[nn.Conv1d(c_in, 1, kernel_size=1), nn.ReLU()])
        
        # Common
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = nn.Linear(hidden_size * (1 + bidirectional) + conv_layers[-1]+seq_len, c_out)
        

    def forward(self, x):  
        # RNN
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)
        
        # FCN
        x1 = self.convblock1(x)
        x1 = self.se1(x1)
        x1 = self.convblock2(x1)
        x1 = self.se2(x1)
        x1 = self.convblock3(x1)
        x1 = self.gap(x1)

        x2 = self.conv2dblock(x)
        x2 = self.conv2d1x1block(x2)
        x2 = self.conv1dblock(x2)
        #print(x2.shape)
       
        # Concat
        #print(x1.shape)
        x = self.concat([last_out, x1, x2.squeeze(1)])
        x = self.fc_dropout(x)
        x = self.fc(x)
        
        return x
    


class LSTM_2dCNN(Module):
    def __init__(self, c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout=0., conv_layers=[128, 256, 128], kss=[7, 5, 3], se=0):
        
        #super().__init__()
        
        if shuffle: assert seq_len is not None, 'need seq_len if shuffle=True'
            
        # RNN
        self.rnn = nn.LSTM(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        # FCN
        assert len(conv_layers) == len(kss)

        #2dCNN
        self.conv2dblock = nn.Sequential(*[Unsqueeze(1), Conv2d(1, conv_layers[0], kernel_size=(1, seq_len), padding='same'), BatchNorm(conv_layers[0]), nn.ReLU()])
        self.conv2d1x1block = nn.Sequential(*[nn.Conv2d(conv_layers[0], 1, kernel_size=1), nn.ReLU(), Squeeze(1)])
        self.conv1dblock = nn.Sequential(*[nn.Conv1d(c_in, 1, kernel_size=1), nn.ReLU()])
        
        # Common
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = nn.Linear(hidden_size * (1 + bidirectional) +seq_len, c_out)
        

    def forward(self, x):  
        # RNN
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)
        

        #2dCNN
        x2 = self.conv2dblock(x)
        x2 = self.conv2d1x1block(x2)
        x2 = self.conv1dblock(x2)
        #print(x2.shape)
       
        # Concat
        #print(x1.shape)
        x = self.concat([last_out, x2.squeeze(1)])
        x = self.fc_dropout(x)
        x = self.fc(x)
        
        return x