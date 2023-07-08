import functools
import inspect
import pickle
import os

import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import numpy as np

import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
from libs.utils import create_folder_if_not_exist


class PlModelInterface(pl.LightningModule):
    def __init__(self,
                 model,
                 loss,
                 lr,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.configure_loss()
        self.init_metrics()

        if 'tune' in self.hparams.keys() and self.hparams['tune']:
            self.best_val_loss = float('inf')

    def init_metrics(self):
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.mse = MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y = batch['decoder_target']

        if self.hparams.loss == 'normal':
            mu, sigma = self(batch)
            loss = self.loss_function(mu, sigma, y)
        else:
            yhat = self(batch)
            loss = self.loss_function(yhat, y)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def log_metrics(self, yhat, y):
        mse = self.mse(yhat, y)
        mae = self.mae(yhat, y)
        mape = self.mape(yhat, y)

        self.log('mse', mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log('mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('mape', mape, on_step=False, on_epoch=True, prog_bar=True)

        return mse, mae, mape

    def validation_step(self, batch, batch_idx):
        y = batch['decoder_target']

        if self.hparams.loss == 'normal':
            yhat, mu, sigma = self.model(batch, num_samples=self.num_samples)
            loss = self.loss_function(mu, sigma, y)
        else:
            yhat = self(batch)
            loss = self.loss_function(yhat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(yhat, y)

        return loss.cpu().numpy()

    def test_step(self, batch, batch_idx):
        y = batch['decoder_target']

        if self.hparams.loss == 'normal':
            yhat, mu, sigma = self.model(batch, num_samples=self.num_samples)
            loss = self.loss_function(mu, sigma, y)
        else:
            yhat = self(batch)
            loss = self.loss_function(yhat, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        mse, mae, mape = self.log_metrics(yhat, y)

        return {
            'yhat': yhat.cpu().numpy(),
            'y': y.cpu().numpy(),
            'loss': loss.cpu().numpy(),
            'mse': mse.cpu().numpy(),
            'mae': mae.cpu().numpy(),
            'mape': mape.cpu().numpy()
        }

    def validation_epoch_end(self, outputs):
        if 'tune' in self.hparams.keys() and self.hparams['tune']:
            cur_val_loss = []
            for out in outputs:
                cur_val_loss.append(out)
            cur_val_loss = float(np.mean(cur_val_loss))
            if cur_val_loss < self.best_val_loss:
                self.best_val_loss = cur_val_loss

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        # self.print('')
        pass

    def test_epoch_end(self, outputs):
        test_save_dir = self.hparams.test_save_dir
        create_folder_if_not_exist(test_save_dir)
        with open(os.path.join(test_save_dir, 'test_outputs.pickle'), 'wb') as f:
            pickle.dump(outputs, f)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'L1':
            self.loss_function = F.l1_loss
        elif loss == 'smooth_L1':
            self.loss_function = F.smooth_l1_loss
        elif loss == 'huber':
            delta = self.hparams.huber_delta
            self.loss_function = functools.partial(F.huber_loss, delta=delta)
        elif loss == 'normal':
            self.loss_function = gaussian_normal_loss
            self.num_samples = self.hparams.num_samples
        else:
            raise ValueError("Invalid Loss Type!")


def gaussian_normal_loss(mu, sigma, y):
    sampler = torch.distributions.Normal(mu, sigma)
    likelihood = sampler.log_prob(y)
    return -torch.mean(likelihood)
