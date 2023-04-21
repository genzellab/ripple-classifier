#%%
import torch
import torch.nn.functional as F
import torch.nn as nn
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
import random
# x = torch.rand(32,64, 2400)
# conv1 = nn.Conv1d(64, 16, kernel_size=3, stride=1, padding=0)
# conv2 = nn.Conv1d(16, 32, kernel_size=6, stride=1, padding=0)
# conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=0)
# conv4 = nn.Conv1d(64, 128, kernel_size=12, stride=2, padding=0)
# conv5 = nn.Conv1d(128, 256, kernel_size=2, stride=2, padding=0)
# fc = nn.Linear(768, 2)
# x = F.relu(conv1(x))
# x = F.relu(conv2(x))
# x = F.relu(conv3(x))
# x = F.relu(conv4(x))
# x = F.relu(conv5(x))
# print(x.shape)

# x = x.view(x.size(0), -1)
# print(x.shape)
# x = fc(x)

# print(x.shape)#%%
df_index = pd.read_csv('proc_data/labeling_data_test_2/data_index.csv')
df_index.columns = ['orig_id', 'mat_key', 'data_idx'   ]
#%%
#%%

df_label_1 = pd.read_csv('proc_data/labeling_data_test_2/labeling_400_ann_1.csv')[['csv','label', 'annotator']]
df_label_1['orig_id'] = df_label_1.csv.apply(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))
df_label_1['annotator'] = 1
df_label_2 = pd.read_csv('proc_data/labeling_data_test_2/labeling_400_ann_2.csv')[[ 'csv','label', 'annotator']]
df_label_2['orig_id'] = df_label_2.csv.apply(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))
df_label_2['annotator'] = 2
df_label_3 = pd.read_csv('proc_data/labeling_data_test_2/labeling_400_ann_3.csv')[[ 'csv','label', 'annotator']]
df_label_3['orig_id'] = df_label_3.csv.apply(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))
df_label_3['annotator'] = 3
# df_label_3 = df_label_3.add_suffix('_annotator_3')zzzzzzzzzzzzzz


# df_label_4 = df_label_4.add_suffix('_annotator_4')
#%%
#merge 400 same examples
df_label = pd.merge(df_label_1, df_label_2, left_on='orig_id', right_on='orig_id', how='inner',)
#merge with df_label_3 and df_label_4
df_label = pd.merge(df_label, df_label_3, left_on='orig_id', right_on='orig_id', how='inner',)
# df_label = pd.merge(df_label, df_label_4, left_on='orig_id', right_on='orig_id', how='outer')
#rename the columns annotator, annotator_x, annotator_y to annotator_1, annotator_2, annotator_3
df_label.columns = ['csv_1', 'label_1', 'annotator_1', 'orig_id', 'csv_2', 'label_2', 'annotator_2', 'csv_3', 'label_3', 'annotator_3']
df_label
#%%
#merge splits of examples
#concatenate df_label_1 and df_label_2
df_label = pd.concat([df_label_1, df_label_2])
#merge with df_label_3 and df_label_4
df_label = pd.merge(df_label, df_label_3, left_on='orig_id', right_on='orig_id', how='outer',)
df_label = pd.merge(df_label, df_label_4, left_on='orig_id', right_on='orig_id', how='outer')
df_label
#merge the columns annotator_y and annotator
df_label['annotator_2'] = df_label.annotator.combine_first(df_label.annotator_y)
df_label['annotator_1'] = df_label.annotator_x

df_label['label_2'] = df_label.label.combine_first(df_label.label_y)
df_label['label_1'] = df_label.label_x

# #%%
# #merge all df_label on orig_id
# df_label = pd.merge(df_label_1, df_label_2, on='orig_id', suffixes=('_annotator_1', '_annotator_2'))
# df_label = pd.merge(df_label, df_label_3, left_on='orig_id', right_on='orig_id_annotator_3')
# df_label
#%%
#merge df_index and df_label_1 on orig_idx and id 
df = pd.merge(df_index, df_label, on='orig_id')

# df.to_csv('proc_data/labeling_data_test_2/test.csv', index=False)
# #%%
# #get df_index with orig_id not in df
# df_db = df_index[~df_index.orig_id.isin(df.orig_id)]
# df_db['threshold_label'] = df_db.mat_key.apply(lambda x:  x.split('_')[1])
# df_db['rat_id'] = df_db.mat_key.apply(lambda x:  x.split('_')[-2])

# df_db
# #%%

# #get 10 samples per rat from df_db balanced by threshold_label
# df_res = df_db.groupby(['threshold_label', 'rat_id']).apply(lambda x: x.sample(12, replace=False,))
# #%%
# #save to csv
# df_res.to_csv('proc_data/labeling_data_test_2/data_index_2.csv', index=False)
# #%%
# #load data_index_2.csv
# df_index_2 = pd.read_csv('proc_data/labeling_data_test_2/data_index_2.csv')
# df_index_2
#%%
#get threshold label from mat_key
df['threshold_label'] = df.mat_key.apply(lambda x:  x.split('_')[1])
#change complex to cswr
df['threshold_label'] = df.threshold_label.apply(lambda x: x.replace('complex', 'cswr'))
#get rat id from mat_key
df['rat_id'] = df.mat_key.apply(lambda x:  x.split('_')[-2])
df

#%%
# #calculate agreement between label_1 and label_2
# df['agreement_1_2'] = df.apply(lambda x: x.label_1 == x.label_2, axis=1)
# df.agreement_1_2.value_counts()
#change cswr-1, cswr-2, cswr-3 to cswr
df['label_1'] = df.label_1.astype(str)

df['label_1'] = df.label_1.apply(lambda x: x.replace('cswr-1', 'cswr'))
df['label_1'] = df.label_1.apply(lambda x: x.replace('cswr-2', 'cswr'))
df['label_1'] = df.label_1.apply(lambda x: x.replace('cswr-3', 'cswr'))
df['label_2'] = df.label_2.astype(str)

df['label_2'] = df.label_2.apply(lambda x: x.replace('cswr-1', 'cswr'))
df['label_2'] = df.label_2.apply(lambda x: x.replace('cswr-2', 'cswr'))
df['label_2'] = df.label_2.apply(lambda x: x.replace('cswr-3', 'cswr'))
#change dtype label_3 to str
df['label_3'] = df.label_3.astype(str)
df['label_3'] = df.label_3.apply(lambda x: x.replace('cswr-1', 'cswr'))
df['label_3'] = df.label_3.apply(lambda x: x.replace('cswr-2', 'cswr'))
df['label_3'] = df.label_3.apply(lambda x: x.replace('cswr-3', 'cswr'))
#%%
df['agreement_all'] = df.apply(lambda x: x.label_1 == x.label_2 and x.label_1 == x.label_3, axis=1)
df.agreement_all.value_counts()

# #calculate agreement between all agreement_all vote and threshold_label
df['agreement_all_threshold'] = df.apply(lambda x: x.agreement_all == x.threshold_label, axis=1)
df.agreement_all_threshold.value_counts()
#get two or more agreement between annotators   
df['agreement_1_2'] = df.apply(lambda x: x.label_1 == x.label_2, axis=1)
df['agreement_1_3'] = df.apply(lambda x: x.label_1 == x.label_3, axis=1)
df['agreement_2_3'] = df.apply(lambda x: x.label_2 == x.label_3, axis=1)
df['agreement_1_2_3'] = df.apply(lambda x: x.agreement_1_2 == True and x.agreement_1_3 == True and x.agreement_2_3 == True, axis=1)

#calculate majority agreement between all annotators in case of disagreement choose two annotators with agreement if none agree discard
df['majority_vote'] = df.apply(lambda x: x.label_1 if x.agreement_1_2_3 == True else x.label_1 if x.agreement_1_2 == True else x.label_1 if x.agreement_1_3 == True else x.label_2 if x.agreement_2_3 == True else 'discarded_agreement', axis=1)

df.majority_vote.value_counts()

df['agreement_majority_threshold'] = df.apply(lambda x: x.majority_vote == x.threshold_label, axis=1)
df.agreement_majority_threshold.value_counts()
#%%
df.to_csv('proc_data/labeling_data_test_2/res_400_labeling_data.csv', index=False)
#%%
df.query('agreement_1_2 == True').label_1.value_counts()#.threshold_label.value_counts()

#%%
df.query('threshold_label == "sw"').majority_vote.value_counts()
#%%
import random
#calculate majority vote between label_1, label_2 in disagreement choose random and in discarded choose non discarded
df['majority_vote'] = df.apply(lambda x: x.label_1 if x.agreement_1_2 == True else x.label_1 if x.label_1 != 'discarded' else x.label_2, axis=1)

#calculate agreement between majority vote and threshold_label
df['agreement_majority_threshold'] = df.apply(lambda x: x.majority_vote == x.threshold_label, axis=1)
df.agreement_majority_threshold.value_counts()

#%%
df.to_csv('proc_data/labeling_data_test_2/labeling_data.csv', index=False)

#%%
df.query('agreement_majority_threshold == True').threshold_label.value_counts()
#%%
df.query('agreement_1_2 == False').majority_vote.value_counts()
df.query('agreement_1_2 == False and majority_vote == "ripple"').sample(5).to_csv('proc_data/labeling_data_test_2/ripple.csv', index=False)
df.query('agreement_1_2 == False and majority_vote == "cswr"').sample(5).to_csv('proc_data/labeling_data_test_2/cswr.csv', index=False)
df.query('agreement_1_2 == False and majority_vote == "sw"').sample(5).to_csv('proc_data/labeling_data_test_2/sw.csv', index=False)
df.query('agreement_1_2 == False and majority_vote == "swr"').sample(5).to_csv('proc_data/labeling_data_test_2/swr.csv', index=False)

#%%
#calculate agreement between threshold_label and label_1 when annotator_1 == 1
df.apply(lambda x: x.threshold_label == x.label_1 if x.annotator_1 == 1 else None, axis=1).value_counts()
#calculate agreement between threshold_label and label_1 when annotator_1 == 2
df.apply(lambda x: x.threshold_label == x.label_1 if x.annotator_1 == 2 else None, axis=1).value_counts()
#calculate agreement between threshold_label and label_2 when annotator_2 == 3
df.apply(lambda x: x.threshold_label == x.label_2 if x.annotator_2 == 3 else None, axis=1).value_counts()
#calculate agreement between threshold_label and label_2 when annotator_2 == 4
df.apply(lambda x: x.threshold_label == x.label_2 if x.annotator_2 == 4 else None, axis=1).value_counts()
#%%
#replace complex in threshold_label with cswr
df['threshold_label'] = df.threshold_label.apply(lambda x: x.replace('complex', 'cswr'))
df_annotations = df[['threshold_label', 'label_annotator_1', 'label_annotator_2', 'label_annotator_3', 'rat_id', 'data_idx']]
#%%
#calculate agreement between annotators
df_annotations['agreement_all'] = df_annotations.apply(lambda x: x.label_annotator_1 == x.label_annotator_2 and x.label_annotator_2 == x.label_annotator_3, axis=1)
df_annotations.agreement_all.value_counts() 
#calculate agreement betweeen at least 2 annotators
df_annotations['agreement_2'] = df_annotations.apply(lambda x: x.label_annotator_1 == x.label_annotator_2 or x.label_annotator_2 == x.label_annotator_3 or x.label_annotator_1 == x.label_annotator_3, axis=1)
df_annotations.agreement_2.value_counts()
#calculate majority vote
df_annotations['majority_vote'] = df_annotations.apply(lambda x: x.label_annotator_1 if x.label_annotator_1 == x.label_annotator_2 else x.label_annotator_2 if x.label_annotator_2 == x.label_annotator_3 else x.label_annotator_1, axis=1)
#calculate agreement between majority vote and threshold label
df_annotations['agreement_majority'] = df_annotations.apply(lambda x: x.majority_vote == x.threshold_label, axis=1)
df_annotations.agreement_majority.value_counts()
#%%
#calculate per annotator agreement with threshold label
df_annotations['agreement_annotator_1'] = df_annotations.apply(lambda x: x.label_annotator_1 == x.threshold_label, axis=1)
df_annotations['agreement_annotator_2'] = df_annotations.apply(lambda x: x.label_annotator_2 == x.threshold_label, axis=1)
df_annotations['agreement_annotator_3'] = df_annotations.apply(lambda x: x.label_annotator_3 == x.threshold_label, axis=1)
df_annotations.agreement_annotator_3.value_counts()
#%%
import torchaudio
conformer = torchaudio.models.Conformer(
     input_dim=64,
     num_heads=4,
     ffn_dim=128,
     num_layers=4,
     depthwise_conv_kernel_size=31,)
#%%
lengths = torch.full((10,),2400)  # (batch,)
input = torch.rand(10, 2400, 64)
output = conformer(input, lengths)
#%%
output[0].shape
#%%
nn.Sequential(*[conv1, conv2, conv3, conv4, fc])
#%%
import h5py
# print attributes
h = h5py.File('proc_data/PFC/dataset_PFCshal_ratID3.hdf5', 'r')
for k in h.attrs.keys():
    print(f"{k} => {h.attrs[k]}")
print(h['x'].shape)
h.close()
# %%
dict(h.attrs.items())

#%%
import wandb
run = wandb.init(project="ripple_project", job_type='dataset')
# %%
my_data = wandb.Artifact("PFC_dataset", type="raw_data",metadata={'sampling_rate': 600, 'band_filtered': True,'length_s':6})
my_data.add_dir("data/PFCshal")
run.log_artifact(my_data)
#%%
my_data = wandb.Artifact("HPC_dataset", type="raw_data",metadata={'sampling_rate': 600, 'band_filtered': True,'length_s':6})
my_data.add_dir("data/HPCpyra")
run.log_artifact(my_data)
# %%
my_data = wandb.Artifact("HPC_dataset", type="raw_data",metadata={'sampling_rate': 600, 'band_filtered': True,'length_s':6})
my_data.add_dir("data/HPCpyra")
run.log_artifact(my_data)
#%%
import h5py
h = h5py.File('proc_data/dataset_HPCpyra_ratID3.hdf5', 'r')
for k in h.attrs.keys():
    print(f"{k} => {h.attrs[k]}")
print(h['x'].shape)
# h.close()


# %%
h.attrs.keys()
#%%
my_data = wandb.Artifact("HPC_preproc", type="preprocessed_data",metadata=dct)
my_data.add_dir("proc_data/proc_hp")
run.log_artifact(my_data)
#%%
dct = {}
for k in h.attrs.keys():
    if k == 'data_types': continue
    dct[k] = h.attrs[k]
#%%
import pandas as pd
import json
label_arr = json.loads(h.attrs['data_types'])
y = pd.Series(h['y']).apply(lambda x: label_arr[x])
dct['n_complex'] = y[y.str.contains('complex')].shape[0]
dct['n_ripple'] = y[y.str.contains('ripple')].shape[0]
dct['n_swr'] = y[~y.str.contains('complex') & y.str.contains('swr')].shape[0]
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
data = pd.DataFrame({'accuracy':[0.52,0.56,0.33,0.37],'experiment':['HPC','HPC+PCA','HPC+PCA_CBD','HPC_CBD']},)
#barplot data and add chance line at 0.33
sns.barplot(data=data,x='experiment',y='accuracy', palette='Blues_d')
plt.axhline(0.33, ls='--', color='grey')
#show actual value of each bar  
for index, row in data.iterrows():
    plt.text(row.name,row.accuracy, round(row.accuracy,2), color='black', ha="center")
plt.ylabel('Accuracy')
plt.xlabel('Dataset')
plt.title('Accuracy of different datasets')
plt.savefig('accuracy_HPC.png', dpi=100)

plt.show()

#save figure
#%%
data = pd.DataFrame({'accuracy':[0.397],'experiment':['Veh_VS_CBD']},)
#barplot data and add chance line at 0.33
sns.barplot(data=data,x='experiment',y='accuracy', palette='Blues_d')
#make plot thinner
plt.gcf().set_size_inches(3, 5)
plt.axhline(0.52, ls='--', color='grey')
#label grey bar chance bar
plt.text(0,0.52, 'zeroR=0.52', color='black', ha="center")
#show actual value of each bar  
for index, row in data.iterrows():
    plt.text(row.name,row.accuracy, round(row.accuracy,2), color='black', ha="center")
plt.ylabel('Accuracy')
plt.xlabel('Dataset')
plt.title('Train on HPC Veh and test on HPC CBD')
plt.savefig('accuracy_HPC_vs.png')

plt.show()
#save figure
#%%
data = pd.DataFrame({'accuracy':[0.373,0.33],'experiment':['PFC','PFC_CBD']},)
#barplot data and add chance line at 0.33
sns.barplot(data=data,x='experiment',y='accuracy', palette='Blues_d')
plt.axhline(0.33, ls='--', color='grey')
#show actual value of each bar  
for index, row in data.iterrows():
    plt.text(row.name,row.accuracy, round(row.accuracy,2), color='black', ha="center")
plt.ylabel('Accuracy')
plt.xlabel('Dataset')
plt.title('Accuracy of different datasets')
plt.savefig('accuracy_PFC.png')

plt.show()
#save figure
# %%
data = pd.DataFrame({'accuracy':[0.4276],'experiment':['Veh_VS_CBD']},)
#barplot data and add chance line at 0.33
sns.barplot(data=data,x='experiment',y='accuracy', palette='Blues_d')
#make plot thinner
plt.gcf().set_size_inches(3, 5)
plt.axhline(0.52, ls='--', color='grey')
#label grey bar chance bar
plt.text(0,0.52, 'zeroR=0.52', color='black', ha="center")
#show actual value of each bar  
for index, row in data.iterrows():
    plt.text(row.name,row.accuracy, round(row.accuracy,2), color='black', ha="center")
plt.ylabel('Accuracy')
plt.xlabel('Dataset')
plt.title('Train on HPC Veh and test on HPC CBD')
plt.savefig('accuracy_PFC_vs.png')

plt.show()
#save figure
#%%

rnn = nn.LSTM(8, 20, 12, batch_first=True,bidirectional=True)
input = torch.randn(64, 90, 8)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input)
# %%
output.shape, hn.shape, cn.shape,hn.view(12, 2, hn.shape[1], hn.shape[2]).shape
# %%
ts = hn.view(12, 2, hn.shape[1], hn.shape[2])
torch.cat((ts[:, 0, :, :], ts[:, 1, :, :]), dim=2).shape
#%%
from scipy.io import loadmat
data = loadmat('data/VEH_HPC_TIMEBINS/GC_ratID3_veh.mat')
# %%
data.keys()
# %%
data['GC_complex_swr_ratID3_veh'][0][0][0][0][0][0]#.shape
pd.DataFrame(data['GC_complex_swr_ratID3_veh'][0][0][0][0][0])
#%%
# %%                                            (1,1),1,(),51,1,4,3601
d = data['GC_complex_swr_ratID3_veh']#.shape
d,d.shape,
d['grouped_oscil_table'][0,0][0][3][1,0]
# %%
data_veh = loadmat('data/VEH_TIMEBINS/GC_ratID3_veh.mat')
data_veh.keys()
data_veh['cr'].flatten().shape
# %%
import pandas as pd
df_cr = pd.DataFrame(data_veh['cr']).astype(int)
df_cr#.T.value_counts()
df_cr.T
#%%
import pandas as pd
df = pd.read_csv('proc_data/HPC_150ms/data_index.csv')
# %%
df.loc[df[(df.rat_id==3) & (df.label==0)].index,'label'] = 1
df.query('rat_id==3 and label==1')
#%%
import torch
from perceiver_pytorch import Perceiver

model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                 #   depth * (cross attention -> self_per_cross_attn * self attention)
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,         # number of dimensions per cross attention head
    latent_dim_head = 64,        # number of dimensions per latent self attention head
    num_classes = 1000,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    self_per_cross_attn = 2      # number of self attention blocks per cross attention
)

img = torch.randn(1, 224, 224, 3) # 1 imagenet image, pixelized

model(img) # (1, 1000)
# %%
import torch
from perceiver_pytorch import Perceiver

model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 1,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                 #   depth * (cross attention -> self_per_cross_attn * self attention)
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,         # number of dimensions per cross attention head
    latent_dim_head = 64,        # number of dimensions per latent self attention head
    num_classes = 3,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    self_per_cross_attn = 2      # number of self attention blocks per cross attention
)

img = torch.randn(1, 90, 3) # 1 imagenet image, pixelized

model(img) # (1, 1000)
#%%
filename = 'data/VEH_FEATURES/GC_ratID3_veh.mat'
ratID = int(filename.split('_')[-2].split('ratID')[-1])
ratID
mat = loadmat(filename)
mat.keys()
#%%
#%%
import os
label_dct = {'complex_swr':0,'swr':1,'ripple':2}
df = pd.DataFrame(columns=['id','rat_id','label','MeanFreq','Amplitude','AUC','Duration','Peak2Peak','Power','Entropy','NumberOfPeaks','SpecEntropy'])

#walk through files in folder
for root, dirs, files in os.walk('data/CBD_FEATURES'):
    for file in files:
        if file.endswith('.mat'):
            filename = os.path.join(root, file)
            print(filename)
            ratID = int(filename.split('_')[-2].split('ratID')[-1])
            mat = loadmat(filename)
            for k in ['complex_swr','swr','ripple']:
                data = mat['GC_' + k + '_ratID'+ str(ratID) +'_cbd']['PCA_features'][0,0]
                df_tmp = pd.DataFrame(data,columns=['MeanFreq','Amplitude','AUC','Duration','Peak2Peak','Power','Entropy','NumberOfPeaks','SpecEntropy'])
                df_tmp['id'] = df_tmp.index
                df_tmp['rat_id'] = ratID
                df_tmp['label'] = label_dct[k]
                df = pd.concat([df,df_tmp])
# %%
df = pd.read_csv('proc_data/veh_features.csv')
#%%
df_train = df[~df.rat_id.isin([206,210])]
df_test = df[df.rat_id.isin([206,210])] 
#%%
X = df_train.drop(columns=['id','rat_id','label']).values
y = df_train['label'].values.astype(int)
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()   
X = scaler.fit_transform(X)
#%% 
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
#%%
#%%
#train a random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500, max_depth=8,random_state=0)
clf.fit(X_train, y_train)

#%%
#test the classifier
y_pred = clf.predict(X_val)
#%%
# classification report
from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))
#%%
#test the classifier
df_test = df[df.rat_id.isin([206,210])] 
min_class_count = df_test.label.value_counts().min()
# get first n samples from each class
df_test = df_test.groupby('label').head(min_class_count)
X_test = df_test.drop(columns=['id','rat_id','label']).values
y_test = df_test['label'].values.astype(int)
X_test = scaler.transform(X_test)
y_pred = clf.predict(X_test)    
print(classification_report(y_test, y_pred))
accuracy_score(y_test, y_pred)

#%%
#hyperparam search for random forest
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'n_estimators': [100, 200, 300, 400, 500]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)

#test the classifier
y_pred = grid_search.predict(X_val)
#%%
# classification report
from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))
grid_search.best_params_
# %%
#use test set
X_test = df_test.drop(columns=['id','rat_id','label']).values
y_test = df_test['label'].values.astype(int)
X_test = scaler.transform(X_test)
y_pred = grid_search.predict(X_test)    
print(classification_report(y_test, y_pred))
#get accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
# %%
#hyperparam search for mlp
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [(100,),(300,),(100,100),(400,400),(500,500)],
    'activation': ['identity', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
# Create a based model
mlp = MLPClassifier(max_iter=1000)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = mlp, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
# %%
#test the classifier
y_pred = grid_search.predict(X_val)
#%%
# classification report
from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))
grid_search.best_params_
# %%
