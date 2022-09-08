# %%
from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import h5py
import numpy as np
from datasets.ripple_spect import RippleSpectDataset
# %%


class RippleSpectMultiDataset(Dataset):
    def __init__(self,
                 data_dir_HPC=None,
                 data_dir_PFC=None,
                 set_type="train",
                 transforms=None,
                 fold=1,
                 num_classes=3,
                 lazy_load=False,
                 ):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.data_dir = data_dir_HPC
        self.data_dir_PFC = data_dir_PFC
        self.num_classes = num_classes
        self.transforms = transforms
        self.lazy_load = lazy_load
        self.hpc_dataset = RippleSpectDataset(data_dir_HPC, set_type, "HPC", transforms, fold, num_classes, lazy_load)
        self.pfc_dataset = RippleSpectDataset(data_dir_PFC, set_type, "PFC", transforms, fold, num_classes, lazy_load)
        
        self.data_df = self.hpc_dataset.data_df
        #check if the dataframes are the same
        assert (self.data_df.label.values == self.pfc_dataset.data_df.label.values).all()
        self.length = len(self.data_df)

        print(self.length, set_type, 'fold', fold)
        print(self.data_df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        hpc_data, label = self.hpc_dataset[idx]
        pfc_data, _ = self.pfc_dataset[idx]

        return (hpc_data,pfc_data), label

# %%
