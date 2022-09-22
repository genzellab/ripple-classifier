import torch
import random

def shift_cutoff(x,signal_size=0.150,max_shift_s=0.015,fs=600):    
    #randomly move the spectrogram +- shift_ms
    shift_p = int(max_shift_s*fs)
    shift_val = random.randint(-shift_p,shift_p)
    x = torch.roll(x,shifts=shift_val,dims=1)
    x_size = x.shape[1]
    #obtain a signal of signal_size
    signal_size_p = int(signal_size*fs)
    x = x[:,x_size//2-signal_size_p//2:x_size//2+signal_size_p//2]    
    return x