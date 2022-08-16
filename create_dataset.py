# %%
from scipy.io import loadmat
import pywt
from argparse import ArgumentParser
import os
import numpy as np
import h5py
import json


def create_wavelet(signal, scales, waveletname, sampling_period):
    """
    Create wavelet from signal
    """
    [coefficients, frequencies] = pywt.cwt(
        signal, scales, waveletname, sampling_period)
    return coefficients


# %%
if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data-loc', type=str,
                        default='data/', help='File location')
    parser.add_argument('--recording-loc', type=str,
                        default='PFC', help='Recording location')
    parser.add_argument('--wavelet-scales', type=int,
                        default=128, help='Number of wavelet scales')
    parser.add_argument('--wavelet-name', type=str,
                        default='cmor1.5-1.0', help='Wavelet name')
    parser.add_argument('--event-window-s', type=int,
                        default=1, help='Event window size in seconds')
    parser.add_argument('--sampling-freq', type=float,
                        default=600, help='Sampling frequency')
    parser.add_argument('--output-loc', type=str,
                        default='datasets/', help='Output location')

    # args = parser.parse_args()
    hparams, _ = parser.parse_known_args()

    y_label_dict = {'complex': 0, 'swr': 1, 'ripple': 2}
    # check if output directory exists
    if not os.path.exists(hparams.output_loc):
        os.makedirs(hparams.output_loc)

    metadata = {'sampling_freq': hparams.sampling_freq, 'event_windows_s': hparams.event_window_s,
                'wavelet_scales': hparams.wavelet_scales, 'wavelet_name': hparams.wavelet_name}
    data_types = []
    # walk through all files in data folder
    for root, dirs, files in os.walk(hparams.data_loc):
        for file in files:
            if file.endswith('.mat') and hparams.recording_loc in file:
                #create h5 file
                f_name = file.split('_')[0]
                h = h5py.File(os.path.join(hparams.output_loc,
                              f'dataset_{f_name}.hdf5'), "w")

                # load data
                data = loadmat(os.path.join(root, file))
                x = None
                y = None
                for data_type in data.keys():
                    if data_type.split('_')[1] in y_label_dict:
                        data_types.append(data_type)
                        data_seq = data[data_type][:, int(
                            hparams.event_window_s*hparams.sampling_freq/2):int(hparams.event_window_s*hparams.sampling_freq)]
                        print(data_type, data_seq.shape)
                        # create wavelet
                        data_seq = create_wavelet(data_seq, np.arange(
                            1, hparams.wavelet_scales), hparams.wavelet_name, hparams.sampling_freq)
                        #store as batch,wavelet,time
                        data_seq = np.transpose(data_seq, (1, 0, 2))
                        data_label = np.full(
                            data_seq.shape[0], y_label_dict[data_type.split('_')[1]])
                        if x is None:
                            x = data_seq
                            y = data_label
                        else:
                            x = np.concatenate((x, data_seq), axis=0)
                            y = np.concatenate((y, data_label), axis=0)
                # save wavelet
                h.create_dataset('x', data=x)
                h.create_dataset('y', data=y)
                h.attrs.update(metadata)
                h.attrs['data_types'] = json.dumps(data_types)
                h.close()
