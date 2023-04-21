#%%
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
event_window_s=6
sampling_freq=600    
output_loc = 'proc_data/labeling_data_test_2/'
data_loc = 'data/VEH_FEATURES_FINAL_2'
# label_issues_df = pd.read_csv('label_issues.csv')
# label_issues_df
df = pd.read_csv('proc_data/labeling_data_test_2/data_index_2.csv')
#rename threshold_label to label
df = df.rename(columns={'threshold_label': 'label'})
#%%
#sample 100 with no replacement
split_1 = df.sample(n=100,replace=False)
split_2 = df[~df.index.isin(split_1.index)].sample(n=100,replace=False)
split_3 = df[~df.index.isin(split_1.index) & ~df.index.isin(split_2.index)].sample(n=100,replace=False)
split_4 = df[~df.index.isin(split_1.index) & ~df.index.isin(split_2.index) & ~df.index.isin(split_3.index)].sample(n=100,replace=False)
#%%
#save all splits
split_1.to_csv(os.path.join(output_loc,'split_1.csv'),index=False)
split_2.to_csv(os.path.join(output_loc,'split_2.csv'),index=False)
split_3.to_csv(os.path.join(output_loc,'split_3.csv'),index=False)
split_4.to_csv(os.path.join(output_loc,'split_4.csv'),index=False)
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
                            label = k.split('_')[1]#y_label_dict[k.split('_')[1]]
                            rat_id = file.split('_')[-2]#int(file.split('_')[-2].split('ratID')[1])
                            if label_issues_df.query('label == @label and rat_id == @rat_id and data_idx == @i').shape[0] == 0:
                                continue
                        df =pd.DataFrame(arr[i]).T
                        df.columns = ['pfcshallow','hpcpyra_filt','hpcbelo_filt','pfcdeep','hpcpyra_raw','hpcbelo_raw']
                        df['time'] = np.linspace(0, event_window_s, arr.shape[2])
                        df.to_csv(os.path.join(output_loc,'data',str(idx_df + i) +'_id.csv'),index=False)
                    print(idx_df)
    data_df.columns = ['mat_key','data_idx']
    data_df.to_csv(os.path.join(output_loc,'data_index.csv'))
create_labeling_dataset(os.path.join(output_loc,'split_4'), data_loc, event_window_s, split_4)    
#%%
df_all= None
for root, dirs, files in os.walk(output_loc):
        for file in files:
            if file.endswith('.csv'):
                if df_all is None:
                    df_all = pd.read_csv(os.path.join(root,file))
                else:
                    df_all = pd.concat([df_all,pd.read_csv(os.path.join(root,file))],ignore_index=True)
df_all.to_csv(os.path.join(output_loc,'all_data.csv'),index=False)
#%%
df_idx = pd.read_csv(os.path.join(output_loc,'data_index.csv'))
df_idx
# %%
df_labels = pd.read_csv('/home/ricardo/Downloads/project-2-at-2022-12-07-13-27-f3726f90.csv')
df_labels.csv = df_labels.csv.apply(lambda x: int(x.split('-')[1].split('_')[0]))
df_labels
# %%
#merge df_idx and df_labels by index and csv
df_idx = df_idx.merge(df_labels, left_on=['Unnamed: 0'], right_on=['csv'])
# %%
#get rows from rat id 201
df_idx[df_idx.mat_key.str.contains("ratID201")].shape#.query('mat_key.str.contains("ratID201")')