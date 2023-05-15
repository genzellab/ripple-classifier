import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from scipy import signal
from scipy.io import savemat
import pandas as pd
from scipy.signal import hilbert, chirp
from scipy.signal import medfilt
from scipy import signal



## Ripple

ripple_belo = pd.read_csv('/home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT/wave_ripple_hpcbelo.csv',
                   sep=',', header=None)
duration_ripple = pd.read_csv('/home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT/duration_ripple.csv',
                   sep=',', header=None)

# # transposed format
# freq = np.transpose(np.array(freq))

ripple_belo = np.array(ripple_belo)
duration_ripple = np.array(duration_ripple)

# # squeeze
# ripple_belo = np.squeeze(ripple_belo)

# Check the shape of data
ripple_belo.shape
duration_ripple.shape


fs = 600
width = 6  # morlet2 width
low_f = 1  # lowest frequency of interest
high_f = 200  # highest frequency of interest
freq = np.linspace(low_f, high_f, num=int(high_f / low_f))  # frequency resolution of time-frequency plot
widths = width * fs / (2 * freq * np.pi)  # wavelet widths for all frequencies of interest

insta_freq_ripple_morlet = np.zeros((1, ripple_belo.shape[0]))
duration_ripple_ = np.zeros((ripple_belo.shape[0],))

for index in range(ripple_belo.shape[0]):

    duration_ripple_[0] = duration_ripple[index, 1] - duration_ripple[index, 0]
    
    segment_index = np.arange(duration_ripple[0, 0], duration_ripple[0, 1])
    
    # HPCbelo
    data_signal = ripple_belo[index, :]
       
    cwtm = signal.cwt(data_signal, signal.morlet2, widths, w=width)
    segment = np.abs(np.sum(cwtm[:, segment_index], axis=1))
    x = np.dot(segment / np.sum(segment), freq)
    insta_freq_ripple_morlet[0, index] = x
    
    
    # plt.plot(abs(cwtm.T))
    # plt.show()

    
features_ripple = np.zeros((ripple_belo.shape[0], 1))
features_ripple = insta_freq_ripple_morlet.T
   

features_ripple.shape

# from np to csv
# convert array into dataframe
DF = pd.DataFrame(features_ripple)
 
# save the dataframe as a csv file
DF.to_csv("features_ripple.csv")






## SW

sw_belo = pd.read_csv('/home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT/wave_sw_hpcbelo.csv',
                   sep=',', header=None)
duration_sw = pd.read_csv('/home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT/duration_sw.csv',
                   sep=',', header=None)

# # transposed format
# freq = np.transpose(np.array(freq))

sw_belo = np.array(sw_belo)
duration_sw = np.array(duration_sw)

# # squeeze
# sw_belo = np.squeeze(sw_belo)

# Check the shape of data
sw_belo.shape
duration_sw.shape


fs = 600
width = 6  # morlet2 width
low_f = 1  # lowest frequency of interest
high_f = 200  # highest frequency of interest
freq = np.linspace(low_f, high_f, num=int(high_f / low_f))  # frequency resolution of time-frequency plot
widths = width * fs / (2 * freq * np.pi)  # wavelet widths for all frequencies of interest

insta_freq_sw_morlet = np.zeros((1, sw_belo.shape[0]))
duration_sw_ = np.zeros((sw_belo.shape[0],))

for index in range(sw_belo.shape[0]):

    duration_sw_[0] = duration_sw[index, 1] - duration_sw[index, 0]
    
    segment_index = np.arange(duration_sw[0, 0], duration_sw[0, 1])
    
    # HPCbelo
    data_signal = sw_belo[index, :]
       
    cwtm = signal.cwt(data_signal, signal.morlet2, widths, w=width)
    segment = np.abs(np.sum(cwtm[:, segment_index], axis=1))
    x = np.dot(segment / np.sum(segment), freq)
    insta_freq_sw_morlet[0, index] = x
    
    
    # plt.plot(abs(cwtm.T))
    # plt.show()

    
features_sw = np.zeros((sw_belo.shape[0], 1))
features_sw = insta_freq_sw_morlet.T
   

features_sw.shape

# from np to csv
# convert array into dataframe
DF = pd.DataFrame(features_sw)
 
# save the dataframe as a csv file
DF.to_csv("features_sw.csv")



## SWR


swr_belo = pd.read_csv('/home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT/wave_swr_hpcbelo.csv',
                   sep=',', header=None)
duration_swr = pd.read_csv('/home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT/duration_swr.csv',
                   sep=',', header=None)

# # transposed format
# freq = np.transpose(np.array(freq))

swr_belo = np.array(swr_belo)
duration_swr = np.array(duration_swr)

# # squeeze
# swr_belo = np.squeeze(swr_belo)

# Check the shape of data
swr_belo.shape
duration_swr.shape


fs = 600
width = 6  # morlet2 width
low_f = 1  # lowest frequency of interest
high_f = 200  # highest frequency of interest
freq = np.linspace(low_f, high_f, num=int(high_f / low_f))  # frequency resolution of time-frequency plot
widths = width * fs / (2 * freq * np.pi)  # wavelet widths for all frequencies of interest

insta_freq_swr_morlet = np.zeros((1, swr_belo.shape[0]))
duration_swr_ = np.zeros((swr_belo.shape[0],))

for index in range(swr_belo.shape[0]):

    duration_swr_[0] = duration_swr[index, 1] - duration_swr[index, 0]
    
    segment_index = np.arange(duration_swr[0, 0], duration_swr[0, 1])
    
    # HPCbelo
    data_signal = swr_belo[index, :]
       
    cwtm = signal.cwt(data_signal, signal.morlet2, widths, w=width)
    segment = np.abs(np.sum(cwtm[:, segment_index], axis=1))
    x = np.dot(segment / np.sum(segment), freq)
    insta_freq_swr_morlet[0, index] = x
    
    
    # plt.plot(abs(cwtm.T))
    # plt.show()

    
features_swr = np.zeros((swr_belo.shape[0], 1))
features_swr = insta_freq_swr_morlet.T
   

features_swr.shape


# from np to csv
# convert array into dataframe
DF = pd.DataFrame(features_swr)
 
# save the dataframe as a csv file
DF.to_csv("features_swr.csv")




## COMPLEX SWR

complex_swr_belo = pd.read_csv('/home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT/wave_complex_swr_hpcbelo.csv',
                   sep=',', header=None)
duration_complex_swr = pd.read_csv('/home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT/duration_complex_swr.csv',
                   sep=',', header=None)

# # transposed format
# freq = np.transpose(np.array(freq))

complex_swr_belo = np.array(complex_swr_belo)
duration_complex_swr = np.array(duration_complex_swr)

# # squeeze
# complex_swr_belo = np.squeeze(complex_swr_belo)

# Check the shape of data
complex_swr_belo.shape
duration_complex_swr.shape


fs = 600
width = 6  # morlet2 width
low_f = 1  # lowest frequency of interest
high_f = 200  # highest frequency of interest
freq = np.linspace(low_f, high_f, num=int(high_f / low_f))  # frequency resolution of time-frequency plot
widths = width * fs / (2 * freq * np.pi)  # wavelet widths for all frequencies of interest

insta_freq_complex_swr_morlet = np.zeros((1, complex_swr_belo.shape[0]))
duration_complex_swr_ = np.zeros((complex_swr_belo.shape[0],))

for index in range(complex_swr_belo.shape[0]):

    duration_complex_swr_[0] = duration_complex_swr[index, 1] - duration_complex_swr[index, 0]
    
    segment_index = np.arange(duration_complex_swr[0, 0], duration_complex_swr[0, 1])
    
    # HPCbelo
    data_signal = complex_swr_belo[index, :]
       
    cwtm = signal.cwt(data_signal, signal.morlet2, widths, w=width)
    segment = np.abs(np.sum(cwtm[:, segment_index], axis=1))
    x = np.dot(segment / np.sum(segment), freq)
    insta_freq_complex_swr_morlet[0, index] = x
    
    
    # plt.plot(abs(cwtm.T))
    # plt.show()

    
features_complex_swr = np.zeros((complex_swr_belo.shape[0], 1))
features_complex_swr = insta_freq_complex_swr_morlet.T
   

features_complex_swr.shape

# from np to csv
# convert array into dataframe
DF = pd.DataFrame(features_complex_swr)
 
# save the dataframe as a csv file
DF.to_csv("features_complex_swr.csv")
