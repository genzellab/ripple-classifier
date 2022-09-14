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
                 lazy_load=False,
                 exp_type='veh',
                 **kwargs
                 ):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transforms = transforms
        self.lazy_load = lazy_load
        self.data_type = data_type
        self.data_df = pd.read_csv(os.path.join(data_dir, "data_index.csv"))
        # each fold corresponds uses one rat as validation and another for testing
        if exp_type == 'veh':
            fold_dict = {
                1: [
                    [206, 210], [206, 210]  # 210
                ],
            }
        else:
            fold_dict = {
                1: [
                    [214, 205], [214, 205]  # 210
                ],
            }
        if set_type == "train":
            self.data_df = self.data_df[~self.data_df["rat_id"].isin(
                fold_dict[fold][0]) & ~self.data_df["rat_id"].isin(
                fold_dict[fold][1])]
        elif set_type == "val":
            self.data_df = self.data_df[self.data_df["rat_id"].isin(
                fold_dict[fold][0])]

        elif set_type == "test":
            self.data_df = self.data_df[self.data_df["rat_id"].isin(
                fold_dict[fold][1])]
            # get the number of samples in the class with the lowest number of samples
            min_class_count = self.data_df.label.value_counts().min()
            # get first n samples from each class
            self.data_df = self.data_df.groupby('label').head(min_class_count)
            print(self.data_df.label.value_counts(), self.data_df)
        if data_type != "all":
            self.data_df = self.data_df[self.data_df.filename.str.contains(
                data_type)]
        self.data_df = self.data_df.reset_index()
        self.data_df = self.data_df.sort_values(by=['rat_id', 'data_idx'])
        self.metadata = None

        if not lazy_load:
            # get rat id
            if self.num_classes == 2:
                # remove examples with y label 0
                self.data_df = self.data_df[self.data_df.label != 0]
                self.data_df = self.data_df.reset_index()
                self.data_df.label[self.data_df.label == 1] = 0
                self.data_df.label[self.data_df.label == 2] = 1
            self.X = None
            self.y = torch.tensor(self.data_df.label.values)
            for filename in self.data_df.filename.unique():
                h = h5py.File(os.path.join(data_dir, filename), 'r')
                if self.metadata is None:
                    self.metadata = dict(h.attrs.items())
                data = np.array(h['x'])
                # label = np.array(h['y'])
                # print(data.shape, label.shape)
                self.X = torch.tensor(data) if self.X is None else torch.cat(
                    (self.X, torch.tensor(data)))
                # self.y = torch.tensor(label) if self.y is None else torch.cat(
                #     (self.y, torch.tensor(label)))
                h.close()
            self.X = self.X.real
            if self.num_classes == 2:
                self.y[self.y == 1] = 0
                self.y[self.y == 2] = 1
            if self.X.dtype == torch.float64:
                self.X = self.X.type(torch.float)
        else:
            if self.num_classes == 2:
                # remove examples with y label 0
                self.data_df = self.data_df[self.data_df.label != 0]
                self.data_df = self.data_df.reset_index()
                self.data_df.label[self.data_df.label == 1] = 0
                self.data_df.label[self.data_df.label == 2] = 1
            self.h_files = {}
            for filename in self.data_df.filename.unique():
                h = h5py.File(os.path.join(data_dir, filename), 'r')
                if self.metadata is None:
                    self.metadata = dict(h.attrs.items())
                self.h_files[filename] = h

            self.y = torch.tensor(self.data_df.label.values)
        self.length = len(self.data_df)

        print(self.length, set_type, 'fold', fold, 'label counts',
              torch.unique(self.y, return_counts=True),'exp_type',exp_type)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        # print(self.leng_df[idx])
        if self.lazy_load:
            row = self.data_df.iloc[idx]
            h = self.h_files[row.filename]
            data = torch.tensor(
                np.array(h['x'][row.data_idx])).real
            if data.dtype == torch.float64:
                data = data.type(torch.float)
        else:
            data = self.X[idx]
        if self.transforms is not None:
            data = self.transforms(data)
        label = self.y[idx]
        return data, label

# %%
