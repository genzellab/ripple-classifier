#%%
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pywt
from argparse import ArgumentParser
import os
import numpy as np
import h5py
import json
import pandas as pd
from yaml import load
event_window_s=6
sampling_freq=600    
output_loc = 'proc_data/labeling_data_test/'
data_loc = 'data/VEH_FEATURES_FINAL'
label_issues_df = pd.read_csv('label_issues.csv')
label_issues_df
#%%
def create_labeling_dataset(output_loc, data_loc, event_window_s, label_issues_df=None):
    data_df = None
    y_label_dict = {'complex': 0, 'swr': 1, 'sw': 2, 'ripple': 3}
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    if not os.path.exists(os.path.join(output_loc,'data')):
        os.makedirs(os.path.join(output_loc,'data'))
    for root, dirs, files in os.walk(data_loc):
        for file in files:
            if file.endswith('.mat'):
                data_mat = loadmat(os.path.join(root,file))
                print(file)
                for k in data_mat.keys():
                    if k.startswith('__') or len(k) < 10: continue
                    print(k)
                    arr = np.array(data_mat[k]['waveforms'][0,0].ravel().tolist())
                    print(arr.shape)
                    if data_df is None:
                        data_df = pd.DataFrame([[k] * len(arr),list(range(len(arr)))]).T
                        idx_df = 0
                    else:
                        idx_df = data_df.index[-1] + 1
                        data_df = pd.concat([data_df,pd.DataFrame([[k] * len(arr),list(range(len(arr)))]).T],ignore_index=True)
                    for i in range(arr.shape[0]):
                        if label_issues_df is not None:
                            label = y_label_dict[k.split('_')[1]]
                            rat_id = int(file.split('_')[-2].split('ratID')[1])
                            if label_issues_df.query('label == @label and rat_id == @rat_id and id == @i').shape[0] == 0:
                                continue
                        df =pd.DataFrame(arr[i]).T
                        df.columns = ['pfcshallow','hpcpyra_filt','hpcbelo_filt','pfcdeep','hpcpyra_raw','hpcbelo_raw']
                        df['time'] = np.linspace(0, event_window_s, arr.shape[2])
                        df.to_csv(os.path.join(output_loc,'data',str(idx_df + i) +'_id.csv'),index=False)
                    print(idx_df)
    data_df.columns = ['mat_key','data_idx']
    data_df.to_csv(os.path.join(output_loc,'data_index.csv'))
create_labeling_dataset(output_loc, data_loc, event_window_s, label_issues_df)    
