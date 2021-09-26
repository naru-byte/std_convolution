from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.data.sssd2019 import SSSD2019Dataset

class SSSD2019DataModule(pl.LightningDataModule):
    def __init__(self, hparams, val_speakers):
        self.hparams = hparams
        self.val_speakers = val_speakers

        dataset = SSSD2019Dataset(path=hparams.data,
                                  mode='train', 
                                  in_channels= hparams.in_channels, 
                                  max_timesteps=hparams.max_timesteps,
                                  val_speakers=["tmp_speaker"],
                                  augmentations=False)

        self.int2label = dataset.int2label
        self.label2int = dataset.label2int
        self.label_list = dataset.label_list

        self.dims = dataset.__getitem__(0)[1].size()
        self.num_classes = len(self.label_list)

    def prepare_data(self, mode, augmentations):
        dataset = SSSD2019Dataset(path=self.hparams.data,
                                  mode=mode,
                                  in_channels=self.hparams.in_channels,
                                  max_timesteps=self.hparams.max_timesteps,
                                  val_speakers=self.val_speakers,
                                  augmentations=augmentations
        )
        return dataset

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.sssd2019_train = self.prepare_data(mode='train', augmentations=True)
            self.sssd2019_val  = self.prepare_data(mode='val', augmentations=False)

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.sssd2019_test = self.prepare_data(mode='test', augmentations=False)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.sssd2019_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.workers)

    def val_dataloader(self):
        return DataLoader(self.sssd2019_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers)

    def test_dataloader(self):
        return DataLoader(self.sssd2019_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers)