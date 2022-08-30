#%%
import wandb
import pandas as pd
import json
import h5py
run = wandb.init(project="ripple_project", job_type='dataset')
#%%

h = h5py.File('proc_data/PFC/dataset_PFCshal_ratID210.hdf5', 'r')
dct = {}
for k in h.attrs.keys():
    if k == 'data_types': continue
    dct[k] = h.attrs[k]
label_arr = json.loads(h.attrs['data_types'])
y = pd.Series(h['y']).apply(lambda x: label_arr[x])
dct['n_complex'] = y[y.str.contains('complex')].shape[0]
dct['n_ripple'] = y[y.str.contains('ripple')].shape[0]
dct['n_swr'] = y[~y.str.contains('complex') & y.str.contains('swr')].shape[0]
dct
#%%
my_data = wandb.Artifact("PFC_preproc", type="preprocessed_data",metadata=dct)
my_data.add_dir("proc_data/PFC")
run.log_artifact(my_data)