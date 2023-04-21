# %%
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

def create_wavelet(signal, scales, waveletname, sampling_period):
    """
    Create wavelet from signal
    """
    [coefficients, frequencies] = pywt.cwt(
        signal, scales, waveletname, sampling_period)
    return coefficients


def create_dataset(hparams):
    y_label_dict = {'complex': 0, 'swr': 1, 'ripple': 2}

    # check if output directory exists
    if not os.path.exists(hparams.output_loc):
        os.makedirs(hparams.output_loc)
    # walk through all files in data folder
    for root, dirs, files in os.walk(hparams.data_loc):
        for file in files:
            if file.endswith('.mat') and hparams.recording_loc in file:
                # create h5 file
                file_name_arr = file.split('_')
                f_name = file.split('_')[0] if len(
                    file_name_arr) == 2 else file_name_arr[0] + '_' + file_name_arr[2].split('.')[0]
                h = h5py.File(os.path.join(hparams.output_loc,
                              f'dataset_{f_name}.hdf5'), "w")
                print(os.path.join(hparams.output_loc,
                                   f'dataset_{f_name}.hdf5'))
                data_types = []

                # load data
                data = loadmat(os.path.join(root, file))
                x = None
                y = None
                for data_type in data.keys():
                    if data_type.split('_')[1] in y_label_dict:

                        data_types.append(data_type)
                        data_length = data[data_type].shape[1]
                        data_seq = data[data_type][:, int(data_length/2 -
                                                          hparams.event_window_s/2*hparams.sampling_freq):int(data_length/2 + hparams.event_window_s/2*hparams.sampling_freq)]

                        print(data_type, 'data_length',
                              data_length, data_seq.shape)
                        # create wavelet
                        data_seq = create_wavelet(data_seq, np.linspace(
                            hparams.wavelet_scales_start, hparams.wavelet_scales_end, hparams.wavelet_scales_num), hparams.wavelet_name, hparams.sampling_freq)
                        # store as batch,wavelet,time
                        data_seq = np.transpose(data_seq, (1, 0, 2))
                        data_label = np.full(
                            data_seq.shape[0], y_label_dict[data_type.split('_')[1]])
                        print(data_seq.shape, data_label.shape)
                        if x is None:
                            x = data_seq
                            y = data_label
                        else:
                            x = np.concatenate((x, data_seq), axis=0)
                            y = np.concatenate((y, data_label), axis=0)
                print('all', x.shape, y.shape)
                # save wavelet
                h.create_dataset('x', data=x)
                h.create_dataset('y', data=y)
                h.attrs.update(hparams.__dict__)
                h.attrs['data_types'] = json.dumps(data_types)
                h.close()
    dct = {}
    dct['n_complex'] = 0    
    dct['n_swr'] = 0
    dct['n_ripple'] = 0
    dct_data_index = {'filename':[],'rat_id':[],'data_idx':[],'label':[]}
    for root, dirs, files in os.walk(hparams.output_loc):
        for file in files:
            if not file.endswith('.hdf5'): continue
            h = h5py.File(os.path.join(hparams.output_loc,file), 'r')
            
            for k in h.attrs.keys():
                if k == 'data_types': continue
                dct[k] = h.attrs[k]
            label_arr = json.loads(h.attrs['data_types'])
            y = pd.Series(h['y']).apply(lambda x: label_arr[x])
            dct['n_complex'] += y[y.str.contains('complex')].shape[0]
            dct['n_ripple'] += y[y.str.contains('ripple')].shape[0]
            dct['n_swr'] += y[~y.str.contains('complex') & y.str.contains('swr')].shape[0]
            dct_data_index['filename'].extend([file]*y.shape[0])
            dct_data_index['rat_id'].extend([file.split('_')[2].split('.')[0][5:]]*y.shape[0])
            dct_data_index['data_idx'].extend(range(y.shape[0]))
            dct_data_index['label'].extend(h['y'])
    df = pd.DataFrame(dct_data_index)   
    #append time bin information to veh
    df['timebins'] = [0]*df.shape[0]
    rat_id = df['rat_id'].unique()
    for r in rat_id:
        if 'VEH' in hparams.data_loc:
            mat_bins = loadmat(os.path.join('data/VEH_TIMEBINS/GC_ratID'+r+'_veh.mat'))
        else:
            mat_bins = loadmat(os.path.join('data/CBD_TIMEBINS/GC_ratID'+r+'_cbd.mat'))

        print(r,mat_bins['cr'].shape, df.loc[df[(df.rat_id==r) & (df.label==0)].index,'timebins'])
        df.loc[df[(df.rat_id==r) & (df.label==0)].index,'timebins'] = mat_bins['cr'].flatten()
        df.loc[df[(df.rat_id==r) & (df.label==1)].index,'timebins'] = mat_bins['swr'].flatten()
        df.loc[df[(df.rat_id==r) & (df.label==2)].index,'timebins'] = mat_bins['r'].flatten()

    df.to_csv(os.path.join(hparams.output_loc,'data_index.csv'),index=False)    


def create_dataset_mat_format(hparams):
    y_label_dict = {'complex': 0, 'swr': 1, 'sw': 2}
    if hparams.recording_loc != 'HPCbelo':
        y_label_dict = {'complex': 0, 'swr': 1, 'ripple': 2}

    # check if output directory exists
    if not os.path.exists(hparams.output_loc):
        os.makedirs(hparams.output_loc)
    # walk through all files in data folder
    recording_loc_dict = {'PFCshallow': 0,'HPCpyra': 1,'HPCbelo': 2,'PFCdeep': 3,}
    for root, dirs, files in os.walk(hparams.data_loc):
        for file in files:
            if file.endswith('.mat'):
                # create h5 file
                file_name_arr = file.split('_')
                f_name = file.split('_')[1] + '_' + hparams.recording_loc
                h = h5py.File(os.path.join(hparams.output_loc,
                              f'dataset_{f_name}.hdf5'), "w")
                print(os.path.join(hparams.output_loc,
                                   f'dataset_{f_name}.hdf5'))
                data_types = []

                # load data
                data_mat = loadmat(os.path.join(root, file))
                x = None
                y = None
                for data_type in data_mat.keys():
                    if '_' in data_type and data_type.split('_')[1] in y_label_dict:
                        data = np.array(data_mat[data_type]['waveforms'][0,0].ravel().tolist())[:,recording_loc_dict[hparams.recording_loc],:]
                        print('data', data.shape)
                        data_types.append(data_type)
                        data_length = data.shape[1]
                        data_seq = data[:, int(data_length/2 -
                                                          hparams.event_window_s/2*hparams.sampling_freq):int(data_length/2 + hparams.event_window_s/2*hparams.sampling_freq)]

                        print(data_type, 'data_length',
                              data_length, data_seq.shape)
                        # create wavelet
                        data_seq = create_wavelet(data_seq, np.linspace(
                            hparams.wavelet_scales_start, hparams.wavelet_scales_end, hparams.wavelet_scales_num), hparams.wavelet_name, hparams.sampling_freq)
                        # store as batch,wavelet,time
                        data_seq = np.transpose(data_seq, (1, 0, 2))
                        data_label = np.full(
                            data_seq.shape[0], y_label_dict[data_type.split('_')[1]])
                        print(data_seq.shape, data_label.shape,y_label_dict[data_type.split('_')[1]])
                        if x is None:
                            x = data_seq
                            y = data_label
                        else:
                            x = np.concatenate((x, data_seq), axis=0)
                            y = np.concatenate((y, data_label), axis=0)
                print('all', x.shape, y.shape,data_types)
                # save wavelet
                h.create_dataset('x', data=x)
                h.create_dataset('y', data=y)
                h.attrs.update(hparams.__dict__)
                #reverse y_label_dict
                h.attrs['data_types'] = json.dumps({v: k for k, v in y_label_dict.items()})
                h.close()
    dct = {}
    dct['n_complex'] = 0    
    dct['n_swr'] = 0
    if hparams.recording_loc != 'HPCbelo':
        dct['n_ripple'] = 0
    else:
        dct['n_sw'] = 0
    dct_data_index = {'filename':[],'rat_id':[],'data_idx':[],'label':[],'data_idx_label':[]}
    for root, dirs, files in os.walk(hparams.output_loc):
        for file in files:
            if not file.endswith('.hdf5'): continue
            h = h5py.File(os.path.join(hparams.output_loc,file), 'r')
            
            for k in h.attrs.keys():
                if k == 'data_types': continue
                dct[k] = h.attrs[k]
            label_arr = json.loads(h.attrs['data_types'])
            y = pd.Series(h['y']).apply(lambda x: label_arr[str(x)])
            dct['n_complex'] += y[y.str.contains('complex')].shape[0]

            if hparams.recording_loc != 'HPCbelo':
                dct['n_ripple'] += y[y.str.contains('ripple')].shape[0]
            else:
                dct['n_sw'] += y[y.str.contains('sw') & ~y.str.contains('swr')].shape[0]
            dct['n_swr'] += y[y.str.contains('swr') & ~y.str.contains('complex')].shape[0]
            dct_data_index['filename'].extend([file]*y.shape[0])
            dct_data_index['rat_id'].extend([file.split('_')[1][5:]]*y.shape[0])
            dct_data_index['data_idx'].extend(range(y.shape[0]))
            dct_data_index['label'].extend(h['y'])
            flg_idx = 0
            y_idx = []

            for i in range(y.shape[0]):
                if i == 0:
                    y_idx.append(flg_idx)
                elif y.iloc[i] != y.iloc[i-1]:
                    flg_idx = 0
                    y_idx.append(flg_idx)
                else:
                    flg_idx += 1
                    y_idx.append(flg_idx)
            dct_data_index['data_idx_label'].extend(y_idx)
    df = pd.DataFrame(dct_data_index)   
    # #append time bin information to veh
    # df['timebins'] = [0]*df.shape[0]
    # rat_id = df['rat_id'].unique()
    # for r in rat_id:
    #     if 'VEH' in hparams.data_loc:
    #         mat_bins = loadmat(os.path.join('data/VEH_TIMEBINS/GC_ratID'+r+'_veh.mat'))
    #     else:
    #         mat_bins = loadmat(os.path.join('data/CBD_TIMEBINS/GC_ratID'+r+'_cbd.mat'))

    #     print(r,mat_bins['cr'].shape, df.loc[df[(df.rat_id==r) & (df.label==0)].index,'timebins'])
    #     df.loc[df[(df.rat_id==r) & (df.label==0)].index,'timebins'] = mat_bins['cr'].flatten()
    #     df.loc[df[(df.rat_id==r) & (df.label==1)].index,'timebins'] = mat_bins['swr'].flatten()
    #     df.loc[df[(df.rat_id==r) & (df.label==2)].index,'timebins'] = mat_bins['r'].flatten()

    df.to_csv(os.path.join(hparams.output_loc,'data_index.csv'),index=False)    

def create_raw_dataset(hparams):
    y_label_dict = {'complex': 0, 'swr': 1, 'ripple': 2}

    # check if output directory exists
    if not os.path.exists(hparams.output_loc):
        os.makedirs(hparams.output_loc)
    # walk through all files in data folder
    for root, dirs, files in os.walk(hparams.data_loc):
        for file in files:
            if file.endswith('.mat') and hparams.recording_loc in file:
                # create h5 file
                file_name_arr = file.split('_')
                f_name = file.split('_')[0] if len(
                    file_name_arr) == 2 else file_name_arr[0] + '_' + file_name_arr[2].split('.')[0]
                h = h5py.File(os.path.join(hparams.output_loc,
                              f'dataset_{f_name}.hdf5'), "w")
                print(os.path.join(hparams.output_loc,
                                   f'dataset_{f_name}.hdf5'))
                data_types = []

                # load data
                data = loadmat(os.path.join(root, file))
                x = None
                y = None
                for data_type in data.keys():
                    if data_type.split('_')[1] in y_label_dict:

                        data_types.append(data_type)
                        data_length = data[data_type].shape[1]
                        data_seq = data[data_type][:, int(data_length/2 -
                                                          hparams.event_window_s/2*hparams.sampling_freq):int(data_length/2 + hparams.event_window_s/2*hparams.sampling_freq)]

                        print(data_type, 'data_length',
                              data_length, data_seq.shape)
                        # # create wavelet
                        # data_seq = create_wavelet(data_seq, np.linspace(
                        #     hparams.wavelet_scales_start, hparams.wavelet_scales_end, hparams.wavelet_scales_num), hparams.wavelet_name, hparams.sampling_freq)
                        # # store as batch,wavelet,time

                        data_seq = np.expand_dims(data_seq, axis=1)
                        data_label = np.full(
                            data_seq.shape[0], y_label_dict[data_type.split('_')[1]])
                        print(data_seq.shape, data_label.shape)
                        if x is None:
                            x = data_seq
                            y = data_label
                        else:
                            x = np.concatenate((x, data_seq), axis=0)
                            y = np.concatenate((y, data_label), axis=0)
                print('all', x.shape, y.shape)
                # save wavelet
                h.create_dataset('x', data=x)
                h.create_dataset('y', data=y)
                h.attrs.update(hparams.__dict__)
                h.attrs['data_types'] = json.dumps(data_types)
                h.close()
    dct = {}
    dct['n_complex'] = 0    
    dct['n_swr'] = 0
    dct['n_ripple'] = 0
    dct_data_index = {'filename':[],'rat_id':[],'data_idx':[],'label':[]}
    for root, dirs, files in os.walk(hparams.output_loc):
        for file in files:
            if not file.endswith('.hdf5'): continue
            h = h5py.File(os.path.join(hparams.output_loc,file), 'r')
            
            for k in h.attrs.keys():
                if k == 'data_types': continue
                dct[k] = h.attrs[k]
            label_arr = json.loads(h.attrs['data_types'])
            y = pd.Series(h['y']).apply(lambda x: label_arr[x])
            dct['n_complex'] += y[y.str.contains('complex')].shape[0]
            dct['n_ripple'] += y[y.str.contains('ripple')].shape[0]
            dct['n_swr'] += y[~y.str.contains('complex') & y.str.contains('swr')].shape[0]
            dct_data_index['filename'].extend([file]*y.shape[0])
            dct_data_index['rat_id'].extend([file.split('_')[2].split('.')[0][5:]]*y.shape[0])
            dct_data_index['data_idx'].extend(range(y.shape[0]))
            dct_data_index['label'].extend(h['y'])
    df = pd.DataFrame(dct_data_index)   
    #append time bin information to veh
    df['timebins'] = [0]*df.shape[0]
    rat_id = df['rat_id'].unique()
    for r in rat_id:
        mat_bins = loadmat(os.path.join('data/VEH_TIMEBINS/GC_ratID'+r+'_veh.mat'))
        print(r,mat_bins['cr'].shape, df.loc[df[(df.rat_id==r) & (df.label==0)].index,'timebins'])
        df.loc[df[(df.rat_id==r) & (df.label==0)].index,'timebins'] = mat_bins['cr'].flatten()
        df.loc[df[(df.rat_id==r) & (df.label==1)].index,'timebins'] = mat_bins['swr'].flatten()
        df.loc[df[(df.rat_id==r) & (df.label==2)].index,'timebins'] = mat_bins['r'].flatten()

    df.to_csv(os.path.join(hparams.output_loc,'data_index.csv'),index=False)    

#%%
if __name__ == '__main__':
    #recording_loc_dict = {'PFCshallow': 0,'HPCpyra': 1,'HPCbelo': 2,'PFCdeep': 3,}

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data-loc', type=str,
                        default='data/VEH_FEATURES_MULTI', help='File location')
    parser.add_argument('--recording-loc', type=str,
                        default='HPCbelo', help='Recording location')
    parser.add_argument('--wavelet-scales-start', type=int,
                        default=16, help='Wavelet scales start value for linspace')
    parser.add_argument('--wavelet-scales-end', type=int,
                        default=256, help='Wavelet scales end value for linspace')
    parser.add_argument('--wavelet-scales-num', type=int,
                        default=32, help='Wavelet scales num. samples value for linspace')
    parser.add_argument('--wavelet-name', type=str,
                        default='morl', help='Wavelet name')
    parser.add_argument('--event-window-s', type=int,
                        default=0.180, help='Event window size in seconds')
    parser.add_argument('--sampling-freq', type=float,
                        default=600, help='Sampling frequency')
    parser.add_argument('--output-loc', type=str,
                        default='proc_data/HPC_VEH_BELO/', help='Output location')

    # args = parser.parse_args()
    hparams, _ = parser.parse_known_args()
    # create_dataset(hparams)
    create_dataset_mat_format(hparams)
    # create_raw_dataset(hparams)

# %%
event_window_s=6
sampling_freq=600    
data_df = None
output_loc = 'proc_data/labeling_data/'
data_loc = 'data/VEH_FEATURES_FINAL'
if not os.path.exists(output_loc):
    os.makedirs(output_loc)
for root, dirs, files in os.walk(data_loc):
    for file in files:
        if file.endswith('.mat'):
            data_mat = loadmat(os.path.join(root,file))
            print(file)
            for k in data_mat.keys():
                if k.startswith('__') or len(k) < 10: continue
                print(k)
                arr = np.array(data_mat[k]['waveforms'][0,0].ravel().tolist())
                if data_df is None:
                    data_df = pd.DataFrame([[k] * len(arr),list(range(len(arr)))]).T
                    idx_df = 0
                else:
                    idx_df = data_df.index[-1] + 1
                    data_df = pd.concat([data_df,pd.DataFrame([[k] * len(arr),list(range(len(arr)))]).T],ignore_index=True)
                for i in range(arr.shape[0]):
                    df =pd.DataFrame(arr[i]).T
                    df.columns = ['pfcshallow','hpcpyra_filt','hpcbelo_filt','pfcdeep','hpcpyra_raw','hpcbelo_raw']
                    df['time'] = np.linspace(0, event_window_s, arr.shape[2])
                    df.to_csv(os.path.join(output_loc,str(idx_df + i) +'_id.csv'),index=False)
                print(idx_df)
data_df.columns = ['filename','data_idx']
data_df.to_csv(os.path.join(output_loc,'data_index.csv'))
#%%
data_df

#%%
                        
# data_mat = loadmat('/home/ricardo/Downloads/GC_ratID3_veh_waveforms.mat')#'data/VEH_FEATURES_MULTI/GC_ratID3_veh.mat')
data_mat = loadmat('data/VEH_FEATURES_MULTI/GC_ratID3_veh.mat')
print(data_mat.keys())
arr = np.array(data_mat['GC_sw_ratID3_veh']['waveforms'][0,0].ravel().tolist())
# arr = np.stack([data_mat['waveforms']['cswr'][0,0]['belo'][0,0],data_mat['waveforms']['cswr'][0,0]['pyr'][0,0]])
# #change dimension order
# arr = np.moveaxis(arr,0,1)

#%%
df =pd.DataFrame(arr[0]).T
df.columns = ['pfcshallow','hpcpyra','hpcbelo','pfcdeep']
df['time'] = np.linspace(0, event_window_s, data_seq.shape[2])
df.to_csv('data_labeling/ratid3_sw.csv',index=False)
#%%
# data_mat['GC_complex_swr_ratID3_veh']['waveforms']['sw']
data = arr
event_window_s=6
sampling_freq=600
data_length = data.shape[2]
data_seq = data[:,:, int(data_length/2 -
                event_window_s/2*sampling_freq):int(data_length/2 + event_window_s/2*sampling_freq) +1]
time = np.linspace(-event_window_s/2, event_window_s/2, data_seq.shape[2])
time.shape
# time = time2
#%%
idx = 0
# plt.plot(arr[idx,0,:])
figure = plt.figure(figsize=(10, 10))
axis = plt.gca()
axis.plot(time,data[idx,1,:]*10,linewidth = 0.5)
axis.plot(time,data[idx,2,:])
plt.ylim([-1000,1000])
plt.xlim([-0.15,0.15])
plt.vlines(0,-3000,3000,'k',linewidth = 0.1,linestyles='-')
# axis.set_ylim([-data_seq[idx,2,:].max()/2,data_seq[idx,2,:].max()/2])
# axis.set_xlim([-0.5,0.5])
# plt.vlines(0,-1000,1000,'k',linewidth = 0.1,linestyles='-')
# plt.plot(arr[idx,3,:])
#%%
data_mat2 = loadmat('data/VEH_FEATURES_MULTI/GC_ratID3_veh.mat')
data_mat2.keys()
arr2 = np.array(data_mat['GC_complex_swr_ratID3_veh']['waveforms'][0,0].ravel().tolist())
#%%
# data_mat['GC_swr_ratID3_veh']['HPCpyra_trace'][0,0].shape#['waveforms'][0,0].ravel().tolist())[:,2,:]



# h = h5py.File('proc_data/HPC_VEH_PYRA/dataset_ratID3_HPCpyra.hdf5', 'r')
# for k in h.attrs.keys():
#     print(f"{k} => {h.attrs[k]}")
# print(h['x'].shape)
# h.close()
# import numpy as np
# from scipy.io import loadmat
# data = loadmat('data/VEH_HPCpyra/HPCpyra_events_ratID3.mat')
# data.keys()
# x = data['HPCpyra_complex_swr_veh']
# np.expand_dims(x, axis=1).shape
# # %%
# import h5py
# f = h5py.File('proc_data/HPC_150ms/dataset_HPCpyra_ratID3.hdf5', 'r') 
# print(f.keys())

# # %%
# f['x'].shape#.keys()
#%%
import scipy.io as sio

rat_ID_veh = [3,4,9,201,203,206,210,211,213]
rat_ID_cbd= [2,5,10,11,204,205,207,209,212,214]
data_path= '/home/ricardo/Downloads/'

rat_number = 0
data_file = 'GC_ratID' + str(rat_ID_veh[rat_number]) + '_veh_waveforms.mat'
data = sio.loadmat(data_path + data_file)

sw_belo = data['waveforms']['sw'][0,0]['belo'][0,0]
sw_pyr = data['waveforms']['sw'][0,0]['pyr'][0,0]

r_belo = data['waveforms']['r'][0,0]['belo'][0,0]
r_pyr = data['waveforms']['r'][0,0]['pyr'][0,0]

swr_belo = data['waveforms']['swr'][0,0]['belo'][0,0]
swr_pyr = data['waveforms']['swr'][0,0]['pyr'][0,0]

cswr_belo = data['waveforms']['cswr'][0,0]['belo'][0,0]
cswr_pyr = data['waveforms']['cswr'][0,0]['pyr'][0,0]

data_file_duration = 'GC_ratID' + str(rat_ID_veh[rat_number]) + '_veh.mat'
duration = sio.loadmat(data_path + data_file_duration)
# %%
sf = 600
time2 = (np.arange(0,3601) - 3600/2 )/sf
N = int(np.sqrt(sw_belo.shape[0]))
figure, axes = plt.subplots(2, 2)
if N >= 10:
    for i in range(1):
        for j in range(1):
            index = i*10+j
            axes[i,j].plot(time2, sw_belo[index, :], 'r', linewidth = 0.5)
            axes[i,j].plot(time2, sw_pyr[index, :] * 10, 'b', linewidth = 0.5)
            axes[i,j].set_ylim([-1000,1000])
            axes[i,j].set_xlim([-0.15,0.15])
            axes[i,j].vlines(0,-3000,3000,'k',linewidth = 0.1,linestyles='-')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            #axes[i,j].set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
figure.set_size_inches([10,10])
plt.show()
#%%
index = 0
plt.plot(time2, sw_belo[index, :], 'r', linewidth = 0.5)
plt.plot(time2, sw_pyr[index, :] * 10, 'b', linewidth = 0.5)
plt.ylim([-1000,1000])
plt.xlim([-0.15,0.15])
plt.vlines(0,-3000,3000,'k',linewidth = 0.1,linestyles='-')
#%%
(sw_pyr[index, :] == arr[0,1,:]).all()