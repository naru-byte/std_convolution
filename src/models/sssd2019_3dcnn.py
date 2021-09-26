from os.path import join

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch import optim

from src.models.cnn3d import CnnModel, CnnDynamicModel, CnnShapeDynamicModel, CnnStdDynamicModel

class SSSD2019Cnn(pl.LightningModule):

    def __init__(self, hparams, num_classes, in_channels, mode='cnn'):
        super().__init__()
        self.save_hyperparameters()
        self.hparams = hparams
        self.in_channels = in_channels
        self.frontend_dims = [int(dim) for dim in hparams.frontend_dims.split(":")]
        self.pooling_layer = [int(layer_num) for layer_num in hparams.pooling_layer.split(":")]

        if mode == 'cnn':
            self.cnn = CnnModel(input_dim=self.in_channels,
                                frontend_dims=self.frontend_dims,
                                pooling_layer=self.pooling_layer
            )
        elif mode == 'DYcnn':
            self.cnn = CnnDynamicModel(input_dim=self.in_channels,
                                       frontend_dims=self.frontend_dims,
                                       pooling_layer=self.pooling_layer,
                                       num_weights=hparams.num_weights
            )
        elif mode == 'S_DYcnn':
            dynamic_layer = [int(layer_num) for layer_num in hparams.dynamic_layer.split(":")]
            self.cnn = CnnShapeDynamicModel(input_dim=self.in_channels,
                                            frontend_dims=self.frontend_dims,
                                            pooling_layer=self.pooling_layer,
                                            num_weights=hparams.num_weights,
                                            dynamic_layer=dynamic_layer
            )            
        elif mode == 'Ss_DYcnn':
            dynamic_layer = [int(layer_num) for layer_num in hparams.dynamic_layer.split(":")]
            self.cnn = CnnStdDynamicModel(input_dim=self.in_channels,
                                            frontend_dims=self.frontend_dims,
                                            pooling_layer=self.pooling_layer,
                                            num_weights=hparams.num_weights,
                                            dynamic_layer=dynamic_layer
            ) 
        else:
            assert False
        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.classifier = nn.Sequential(nn.Linear(self.frontend_dims[-1], num_classes))

        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.best_val_accuracy = 0.0
        self.best_val_loss = 1000.0

    def forward(self, x, y):
        x = self.cnn(x)
        x = torch.squeeze(self.gmp(x))
        x = self.classifier(x)
        loss, accuracy = self.criterion(x,y), self.accuracy_metrics(x, y)
        return loss, accuracy

    def accuracy_metrics(self, x, y):
        x = torch.argmax(self.softmax(x), dim=1)
        accuracy = torch.mean(torch.eq(x,y).float())
        return accuracy

    def training_step(self, batch, batch_idx):
        img_name, x, y = batch
        loss, accuracy = self.forward(x, y)
        logs = {'train_loss': loss, 'train_accuracy': accuracy}
        self.logger.log_metrics(logs)
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        img_name, x, y = batch
        loss, accuracy = self.forward(x, y)
        logs = {'val_loss': loss, 'val_accuracy': accuracy}
        return {'loss': loss, 'log': logs}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['log']['val_accuracy'] for x in outputs]).mean()

        if self.best_val_accuracy < accuracy:
            self.best_val_accuracy = accuracy
        if self.best_val_loss > loss:
            self.best_val_loss = loss
        
        logs = {
            'val_loss': loss,
            'val_accuracy': accuracy,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
        }

        torch.save(self.state_dict, join(self.hparams.checkpoint_dir,"last_state.ckpt"))
        self.logger.log_metrics(logs)
        return {
            'val_loss': loss,
            'val_accuracy': accuracy,
            'log': logs,
        }

    def test_step(self, batch, batch_idx):
        img_name, x, y = batch
        loss, accuracy = self.forward(x, y)
        logs = {'test_loss': loss, 'test_accuracy': accuracy}
        return {'loss': loss, 'log': logs}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['log']['test_accuracy'] for x in outputs]).mean()
        
        logs = {
            'test_loss': loss,
            'test_accuracy': accuracy,
        }
        
        self.logger.log_metrics(logs)
        print(f'test_loss : {loss}, test_accuracy : {accuracy}')
        return {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'log': logs,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

class SSSD2019DYCnn(SSSD2019Cnn):
    epochs_num = 0
    def __init__(self, hparams, num_classes, in_channels, epochs_num=0):
        super().__init__(hparams, num_classes, in_channels, mode='DYcnn')
        self.epochs_num = epochs_num

    def forward(self, x, y):
        x = self.cnn(x, self.epochs_num)
        x = torch.squeeze(self.gmp(x))
        x = self.classifier(x)
        loss, accuracy = self.criterion(x,y), self.accuracy_metrics(x, y)
        return loss, accuracy

    def training_epoch_end(self, training_step_outputs):
        self.epochs_num += 1

class SSSD2019ShapeDYCnn(SSSD2019Cnn):
    epochs_num = 0
    def __init__(self, hparams, num_classes, in_channels, epochs_num=0):
        super().__init__(hparams, num_classes, in_channels, mode='S_DYcnn')
        self.epochs_num = epochs_num

    def forward(self, x, y):
        x = self.cnn(x, self.epochs_num)
        x = torch.squeeze(self.gmp(x))
        x = self.classifier(x)
        loss, accuracy = self.criterion(x,y), self.accuracy_metrics(x, y)
        return loss, accuracy

    def training_epoch_end(self, training_step_outputs):
        self.epochs_num += 1

class SSSD2019StdDYCnn(SSSD2019Cnn):
    epochs_num = 0
    def __init__(self, hparams, num_classes, in_channels, epochs_num=0):
        super().__init__(hparams, num_classes, in_channels, mode='Ss_DYcnn')
        self.epochs_num = epochs_num

    def forward(self, x, y):
        x = self.cnn(x, self.epochs_num)
        x = torch.squeeze(self.gmp(x))
        x = self.classifier(x)
        loss, accuracy = self.criterion(x,y), self.accuracy_metrics(x, y)
        return loss, accuracy

    def training_epoch_end(self, training_step_outputs):
        self.epochs_num += 1