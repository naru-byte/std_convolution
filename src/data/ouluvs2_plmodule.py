from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.data.ouluvs2 import OuluVS2Dataset

class OuluVS2DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        self.hparams = hparams

        dataset = OuluVS2Dataset(path=hparams.data,
                                 mode='train',
                                 in_channels=hparams.in_channels,
                                 data_mode=hparams.data_mode,
                                 max_timesteps=hparams.max_timesteps,
                                 augmentations=False)

        self.int2label = dataset.int2label
        self.label2int = dataset.label2int
        self.label_list = dataset.label_list

        self.dims = dataset.__getitem__(0)[1].size()
        self.num_classes = len(self.label_list)

    def prepare_data(self, mode, augmentations):
        dataset = OuluVS2Dataset(path=self.hparams.data,
                                 mode=mode,
                                 in_channels=self.hparams.in_channels,
                                 max_timesteps=self.hparams.max_timesteps,
                                 data_mode=self.hparams.data_mode,
                                 augmentations=augmentations
        )
        return dataset

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.ouluvs2_train = self.prepare_data(mode='train', augmentations=True)
            self.ouluvs2_val  = self.prepare_data(mode='val', augmentations=False)

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.ouluvs2_test = self.prepare_data(mode='test', augmentations=False)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.ouluvs2_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.workers)

    def val_dataloader(self):
        return DataLoader(self.ouluvs2_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers)

    def test_dataloader(self):
        return DataLoader(self.ouluvs2_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers)