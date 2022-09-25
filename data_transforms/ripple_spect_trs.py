import torch
import random

class SpectShiftCutoff(object):
    """Shift and cutoff the spectrogram in time and frequency.
    Args:
        shift (int): how much to shift the spectrogram in time
        cutoff (int): how much to cutoff the spectrogram in frequency
    """

    def __init__(self, signal_size=0.150,max_shift_s=0.015,fs=600):
        self.signal_size = signal_size
        self.max_shift_s = max_shift_s
        self.fs = fs

    def __call__(self, x):
        #randomly move the spectrogram +- shift_ms
        shift_p = int(self.max_shift_s*self.fs)
        shift_val = random.randint(-shift_p,shift_p)
        x = torch.roll(x,shifts=shift_val,dims=1)
        x_size = x.shape[1]
        #obtain a signal of signal_size
        signal_size_p = int(self.signal_size*self.fs)
        x = x[:,x_size//2-signal_size_p//2:x_size//2+signal_size_p//2]    
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(signal_size={0}, cutoff={1})'.format(self.signal_size, self.max_shift_s)
