# %%
from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import h5py
import numpy as np

# %%


class RippleSpectDataset(Dataset):
    def __init__(self,
                 data_dir,
                 set_type="train",
                 data_type="HPC",
                 transforms=None,
                 fold=1,
                 num_classes=3,
                 ):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transforms = transforms
        self.data_df = pd.DataFrame(os.listdir(
            data_dir), columns=['filename'])
        if data_type == "HPC":
            self.data_df = self.data_df[self.data_df.filename.str.contains(
                'HPC')]
        elif data_type == "PFC":
            self.data_df = self.data_df[self.data_df.filename.str.contains(
                'PFC')]
        # get rat id
        self.data_df['rat_id'] = self.data_df.filename.apply(
            lambda x: int(x.split('_')[-1].split('.')[0].split('ratID')[1]))

        # each fold corresponds uses one rat as validation and another for testing
        fold_dict = {
            1: [
                [206,210], [206,210]#210
            ],
            # 2: [
            #     210, 206
            # ],
            # 3: [
            #     206, 3
            # ],
            # 4: [
            #     3, 203
            # ],
            # 5: [
            #     203, 211
            # ],
            # 6: [
            #     211, 9
            # ],
            # 7: [
            #     9, 213
            # ],
            # 8: [
            #     213, 4
            # ],
            # 9: [
            #     4, 201
            # ],

        }
        if set_type == "train":
            self.data_df = self.data_df[~self.data_df["rat_id"].isin(
                fold_dict[fold][0]) & ~self.data_df["rat_id"].isin(
                fold_dict[fold][1])]
        elif set_type == "val":
            self.data_df = self.data_df[self.data_df["rat_id"].isin(
                fold_dict[fold][0])]
            print(fold_dict[fold][0])
            print(self.data_df)
        elif set_type == "test":
            self.data_df = self.data_df[self.data_df["rat_id"].isin(
                fold_dict[fold][1])]
        if data_type != "all":
            self.data_df = self.data_df[self.data_df.filename.str.contains(
                data_type)]
            print(self.data_df)
        self.data_df = self.data_df.reset_index()

        self.X = None
        self.y = None
        self.metada = None
        for idx, row in self.data_df.iterrows():
            h = h5py.File(os.path.join(data_dir, row.filename), 'r')
            if self.metada is None:
                self.metada = dict(h.attrs.items())
            data = np.array(h['x'])
            label = np.array(h['y'])
            print(data.shape, label.shape)  
            self.X = torch.tensor(data) if self.X is None else torch.cat(
                (self.X, torch.tensor(data)))
            self.y = torch.tensor(label) if self.y is None else torch.cat(
                (self.y, torch.tensor(label)))
            h.close()
        self.X = self.X.real
        if self.num_classes == 2:
            #remove examples with y label 0
            self.X = self.X[self.y != 0]
            self.y = self.y[self.y != 0]
            self.y[self.y == 1] = 0
            self.y[self.y == 2] = 1
        print(self.X.dtype)
        if self.X.dtype == torch.float64:
            self.X = self.X.type(torch.float)
        self.length = self.X.shape[0]

        print(self.length, set_type, 'fold', fold,'label counts',torch.unique(self.y,return_counts=True))
        print(self.data_df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        # print(self.leng_df[idx])
        data = self.X[idx]
        if self.transforms is not None:
            data = self.transforms(data)
        label = self.y[idx]
        return data, label
