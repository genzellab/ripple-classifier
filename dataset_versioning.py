#%%
import wandb
import pandas as pd
import json
import h5py
run = wandb.init(project="ripple_project", job_type='dataset')
#%%
#load files from directory
import os
data_dir = 'proc_data/VEH_HPC_PCA_180ms'#'proc_data/PFC_128ft'
dct = {}
dct['n_complex'] = 0    
dct['n_swr'] = 0
dct['n_ripple'] = 0
dct_data_index = {'filename':[],'rat_id':[],'data_idx':[],'label':[]}
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if not file.endswith('.hdf5'): continue
        h = h5py.File(os.path.join(data_dir,file), 'r')
        
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
dct       
#%%
df = pd.DataFrame(dct_data_index)
df.to_csv(os.path.join(data_dir,'data_index.csv'),index=False)    

#%%
my_data = wandb.Artifact("HPC_PCA_preproc", type="preprocessed_data",metadata=dct)
my_data.add_dir(data_dir)
run.log_artifact(my_data)
