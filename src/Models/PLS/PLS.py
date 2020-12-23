import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from .PLS_buildingblocks import DSConv3D, DrdbBlock3D, DecoderBlock


class PLS(pl.LightningModule):
    def __init__(self, hparams):
        super(PLS, self).__init__()
        self.hparams = hparams
        self.n_channels = 1
        self.n_classes = 2
        self.current_fold = 0

        # Network specific arguments
        self.nb_channels = [0, 16, 64, 128]
        self.growth_rate = 12

        # ENCODER
        self.ds_conv_1 = DSConv3D(self.nb_channels[0] + 1, self.nb_channels[1])
        self.drdb_1 = DrdbBlock3D(self.nb_channels[1] + 1, self.nb_channels[1] + 1, self.growth_rate)

        self.ds_conv_2 = DSConv3D(self.nb_channels[1] + 1, self.nb_channels[2])
        self.drdb_2_1 = DrdbBlock3D(self.nb_channels[2] + 1, self.nb_channels[2] + 1, self.growth_rate)
        self.drdb_2_2 = DrdbBlock3D(self.nb_channels[2] + 1, self.nb_channels[2] + 1, self.growth_rate)

        self.ds_conv_3 = DSConv3D(self.nb_channels[2] + 1, self.nb_channels[3])
        self.drdb_3_1 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)
        self.drdb_3_2 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)
        self.drdb_3_3 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)
        self.drdb_3_4 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)

        # DECODER
        self.ds_bridge_l2 = DSConv3D(in_chans=self.nb_channels[2] + 1, out_chans=self.n_classes * 2, dstride=1)
        self.ds_bridge_l1 = DSConv3D(in_chans=self.nb_channels[1] + 1, out_chans=self.n_classes * 2, dstride=1)

        self.decoder_l3 = DecoderBlock(in_chans=self.nb_channels[-1] + 1, out_chans=self.n_classes * 2)
        self.decoder_l2 = DecoderBlock(in_chans=self.n_classes * 4, out_chans=self.n_classes * 2)
        self.decoder_l1 = DecoderBlock(in_chans=self.n_classes * 4, out_chans=self.n_classes * 2)

        # OUTPUT
        self.decoder_l0 = nn.Conv3d(in_channels=self.n_classes * 2, out_channels=self.n_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # ENCODER
        # l = 1
        x = x
        input = x
        out = self.ds_conv_1(x)
        downsampled_1 = F.interpolate(input, scale_factor=0.5, mode='trilinear', align_corners=False)
        out = torch.cat([out, downsampled_1], 1)
        out_l1 = self.drdb_1(out)

        # l = 2
        out = self.ds_conv_2(out_l1)
        downsampled_2 = F.interpolate(input, scale_factor=0.25, mode='trilinear', align_corners=False)
        out = torch.cat([out, downsampled_2], 1)
        out = self.drdb_2_1(out)
        out_l2 = self.drdb_2_2(out)

        # l = 3
        out = self.ds_conv_3(out_l2)
        downsampled_3 = F.interpolate(input, scale_factor=0.125, mode='trilinear', align_corners=False)
        out = torch.cat([out, downsampled_3], 1)
        out = self.drdb_3_1(out)
        out = self.drdb_3_2(out)
        out = self.drdb_3_3(out)
        out = self.drdb_3_4(out)

        # DECODER
        out = self.decoder_l3(out)
        out = torch.cat([out, self.ds_bridge_l2(out_l2)], 1)
        out = self.decoder_l2(out)
        out = torch.cat([out, self.ds_bridge_l1(out_l1)], 1)
        out = self.decoder_l1(out)
        out = self.decoder_l0(out)
        out = self.softmax(out)

        return out

#     def training_step(self, batch, batch_nb):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = compute_dice_loss(y_hat, y)
#         tensorboard_logs = {'train_loss': loss}
#         return {'loss': loss, 'log': tensorboard_logs}
#
#     def validation_step(self, batch, batch_nb):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = compute_dice_loss(y_hat, y)
#         return {'val_loss': loss}
#
#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         tensorboard_logs = {'val_loss': avg_loss}
#         return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-8)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
#         return [optimizer], [scheduler]
#
#     def __dataloader(self):
#         train_ds = DirDatasetFolds(self.dataset_path, train=True, augment=self.augment,
#                                    cross_validation_file=self.cross_val_file, fold=self.current_fold)
#         val_ds = DirDatasetFolds(self.dataset_path, val=True, augment=False,
#                                  cross_validation_file=self.cross_val_file, fold=self.current_fold)
#         train_loader = DataLoader(train_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True, shuffle=True)
#         val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True, shuffle=False)
#
#         return {
#             'train': train_loader,
#             'val': val_loader,
#         }
#
#     @pl.data_loader
#     def train_dataloader(self):
#         return self.__dataloader()['train']
#
#     @pl.data_loader
#     def val_dataloader(self):
#         return self.__dataloader()['val']
