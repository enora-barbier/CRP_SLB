

__all__ = ['Marker_NSin', 'Marker_2', 'Marker_3', 'Marker_noise', 'gradcam_simdataset','plot_grad_overview', 'plot_grad_detail','plot_batch', 'Grad_2dCNN', 'Grad_RNN']


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn




def Marker_NSin(z: float, wsize: int):
    """
    Output is a Noisy sign signal with a time sequence
    Input: 
    - z: It is a float or int value in seconds, which will be considered midpoint of the output signal.
    - wsize: Window size of the signal to be generated

    """

    dseq = list(range(int(z - (wsize/2)),int(z + (wsize/2)+1))) #depth the sequence
    f = np.linspace(-np.pi, np.pi, wsize+1)
    a = np.random.random(wsize+1)
    sig = 10*np.sin(f) + a*0.5
    #lat = np.full(wsize+1, x)
    #long = np.full(wsize+1, y)
    X = np.stack((sig, dseq))
    Y = 1
    return X, Y

def Marker_2(z: float, wsize: int):
    
    """
    Output is a Noisy sign signal till half way and a noisy straing line for other half with a time sequence
    Input: 
    - z: It is a float or int value in seconds, which will be considered midpoint of the output signal.
    - wsize: Window size of the signal to be generated
    """

    dseq = list(range(int(z - (wsize/2)),int(z + (wsize/2)+1))) #depth the sequence
    f = np.linspace(-np.pi, 0, int(wsize/2))
    a = np.random.random(int(wsize/2))
    l = np.random.randint(low = 0, high = 2, size = 51)
    sig_ = 10*np.sin(f) + a*0.5
    sig = np.concatenate((sig_, l))
    #lat = np.full(wsize+1, x)
    #long = np.full(wsize+1, y)
    X = np.stack((sig, dseq)) 
    Y = 2
    return X, Y

def Marker_3(z: float, wsize: int):

    """
    Output is a straig line with noise followed by Noisy sign signal and with a time sequence
    Input: 
    - z: It is a float or int value in seconds, which will be considered midpoint of the output signal.
    - wsize: Window size of the signal to be generated
    
    """

    dseq = list(range(int(z - (wsize/2)),int(z + (wsize/2)+1))) #depth the sequence
    f = np.linspace(-np.pi, 0, int(wsize/2))
    a = np.random.random(int(wsize/2))
    l = np.random.randint(low = 1, high = 4, size = 51)
    sig_ = 10*np.sin(f) + a
    sig = np.concatenate(( l, sig_,))
    # lat = np.full(wsize+1, x)
    # long = np.full(wsize+1, y)
    X = np.stack((sig, dseq)) #, lat, long
    Y = 3
    return X, Y
    
def Marker_noise( z: float, wsize: int):

    """
    Output is a Noisy gaussian sign with a time sequence
    Input: 
    - z: It is a float or int value in seconds, which will be considered midpoint of the output signal.
    - wsize: Window size of the signal to be generated
    
    """

    dseq = list(range(int(z - (wsize/2)),int(z + (wsize/2)+1)))
    sig = np.random.randint(low = 1, high = 3, size = wsize+1)
    #lat = np.full(wsize+1, x)
    #long = np.full(wsize+1, y)
    X = np.stack((sig, dseq))
    Y = 0
    return X,Y

#function to creat X and Y sets

def gradcam_simdataset(wsize):

    X = np.empty(shape = (0,2,wsize+1))
    Y = np.empty(shape = (0))
    
    for i in range(0, 1000):
        
        #x = df_loc.iloc[i][1]
        #y = df_loc.iloc[i][2]
        #call noise at time from 1500 to 1600
        dseq_n = np.random.randint(low = 1500, high = 1510, size = 1)
        X_sig, y_sig = Marker_noise(dseq_n, wsize)
        Y = np.append(Y, [y_sig], axis = 0)
        X = np.append(X, [X_sig], axis = 0)

        
        #call marker from random time between 100 to 1000
        dseq_m = np.random.randint(low = 100, high = 1000, size = 1)
        X_sig, y_sig = Marker_NSin( dseq_m, wsize)
        Y = np.append(Y, [y_sig], axis = 0)
        X = np.append(X, [X_sig], axis = 0)

        #call marker from random time between 100 to 1000
        dseq_m = np.random.randint(low = 100, high = 1000, size = 1)
        X_sig, y_sig = Marker_2( dseq_m, wsize)
        Y = np.append(Y, [y_sig], axis = 0)
        X = np.append(X, [X_sig], axis = 0)

    
    return X, Y


def plot_grad_overview(r1):
    
    fig = plt.figure(figsize=(20,10))
    ax = plt.axes()
    plt.title('Observed variables')
    if r1.ndim == 3:
        r1 = r1.mean(0)
        print(r1.shape)
    im = ax.imshow(r1, cmap='seismic')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    im.set_clim(-1,1)



def plot_grad_detail(x, att):

    """
    Input:
    - x : is a single input signal(when batch size is more than 1) 
    - att: the attention output for the imput sub sample of the batch 
    """
    att= (att - att.min()) / (att.max() - att.min())
    x_feature = x.shape[0]
    fig = plt.figure(figsize=(10, 5))
    ax_t = plt.axes()
    for i in range(0, x_feature):
    
        ax1 = plt.subplot(x_feature,1,i+1)
        im = ax1.imshow(att[i].unsqueeze(0), cmap='seismic', aspect='auto' )
        ax1.axis('off')
        im.set_clim(-1,1)
        ax2 = ax1.twinx()
        plt.plot(np.arange(0,101,1), x[i])
        
    
    #print(ax.get_position().x1+0.01 , ax.get_position().y0, 2, ax.get_position().height)
    cax = fig.add_axes([ax1.get_position().x1+0.06 , ax1.get_position().y0, 0.02, ax_t.get_position().height])
    plt.colorbar(im, cax)


def plot_batch(yb):

    """
    yb is the actual output for the batch used for grad-cam
    """
    fig = plt.figure(figsize=(20,10))
    ax = plt.axes()
    plt.title('Batch categories')
    im = ax.imshow(yb.unsqueeze(0))
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)



### CNN hook for Grad_cam

def Grad_2dCNN(model, xb):

    """
    model: The deep learning model to be tested
    xb: the input tensor for which grad_cam is to be performed
    """
    activation_cnn = {}

    def cnn_get_forward(name):
        def hook(model, input, output):
            #print(output)
            activation_cnn[name] = output.detach()
        return hook

    def get_backward(name):
        def hook(model, grad_in, grad_out):
            #print(output)
            activation_cnn[name] = grad_out[0].detach()
        
        return hook
    
    model.conv2dblock.register_forward_hook(cnn_get_forward('forward'))
    model.conv2dblock.register_backward_hook(get_backward('backward'))
    out = model(xb)
    loss= out.mean()
    loss.backward()
    output, grads = activation_cnn['forward'], activation_cnn['backward']
    A_k = output.data
    w_ck = grads.data
    dim = (0, 2, 3) if A_k.ndim == 4 else (0, 2)
    w_ck = torch.neg(w_ck.mean(dim, keepdim=True))
    print(A_k.shape, w_ck.shape)
    L_c = (w_ck * A_k).sum(1)
    L_c = nn.ReLU()(L_c)
    print(L_c.shape)
    if L_c.ndim == 3:  
        L_c = L_c.squeeze(0) if L_c.shape[0] == 1 else L_c
    L_c_cnn = L_c

    return L_c_cnn


### RNN hook


def Grad_RNN(model, xb):

    activation_rnn={}

    def rnn_get_forward(name):
        def hook(model, input, output):
            activation_rnn[name] = output[0].detach()
        return hook

    def get_backward(name):
        def hook(model, grad_in, grad_out):
            #print(output)
            activation_rnn[name] = grad_out[0].detach()
            
        return hook
    
    model.rnn.register_forward_hook(rnn_get_forward('forward'))
    model.rnn.register_backward_hook(get_backward('backward'))
    out = model(xb)
    loss= out.mean()
    loss.backward()
    output, grads = activation_rnn['forward'], activation_rnn['backward']
    A_k = output.data
    w_ck = grads.data
    if A_k.shape[1] > 1:
        A_k = torch.swapaxes(A_k, 1, 2)
        w_ck = torch.swapaxes(w_ck, 1, 2 )
    dim = (0, 2, 3) if A_k.ndim == 4 else (1)
    print(dim)
    w_ck = torch.neg(w_ck.mean(dim, keepdim=True))
    print(grads.shape)
    print(A_k.shape, w_ck.shape)
    L_c = (w_ck * A_k).sum(1)
    L_c = nn.ReLU()(L_c)
    print(L_c.shape)
    if L_c.ndim == 3:  
        L_c = L_c.squeeze(0) if L_c.shape[0] == 1 else L_c
    L_c_rnn = L_c

    return L_c_rnn



