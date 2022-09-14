import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms
from datasets.ripple_spect import RippleSpectDataset
from create_dataset import create_dataset

class RippleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", transforms: transforms.Compose = None, num_workers: int = 4, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        if 'hpc_data_type' in kwargs:
            self.hparams.data_type = kwargs['hpc_data_type']
        else:
            self.hparams.data_type = 'PFC'
        self.transforms = transforms
        self.create_dataset = True

    def prepare_data(self):
        # download
        if self.create_dataset and self.hparams.data_type == 'HPC':
            self.hparams.wavelet_name = self.hparams.wavelet_name + str(self.hparams.wavelet_b) + '-' + str(self.hparams.wavelet_c)
            create_dataset(self.hparams)
            self.create_dataset = False

    def setup(self, stage=None):
        print(self.hparams)
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = RippleSpectDataset(
                self.data_dir, set_type="train", data_type=self.hparams.data_type, transforms=self.transforms, fold=self.hparams.fold, num_classes=self.hparams.num_classes, lazy_load=self.hparams.lazy_load)
            self.val_dataset = RippleSpectDataset(
                self.data_dir, set_type="val", data_type=self.hparams.data_type, transforms=self.transforms, fold=self.hparams.fold, num_classes=self.hparams.num_classes, lazy_load=self.hparams.lazy_load)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = RippleSpectDataset(
                self.data_dir, set_type="test", data_type=self.hparams.data_type, transforms=self.transforms, fold=self.hparams.fold, num_classes=self.hparams.num_classes, lazy_load=self.hparams.lazy_load)

        if stage == "predict" or stage is None:
            self.predict_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
            timeout=300,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
            timeout=300,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
            timeout=300,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
            timeout=300,
        )
