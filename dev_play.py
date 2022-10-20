#%%
import torch
import torch.nn.functional as F
import torch.nn as nn
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

print(x.shape)
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