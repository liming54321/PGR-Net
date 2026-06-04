import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from typing import Dict, List, Tuple, Union, Optional

from easytorch import Runner
from easytorch.core.data_loader import build_data_loader

from ..models.builder import build_model
from iqa import PSNR, SSIM


class BaseImageRunner(Runner):
    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)

        self.cfg = cfg

        self.build_loss(cfg['TRAIN']['LOSS'])

        self.video_bits = cfg['OPTION']['VIDEO_BITS']

        self.metrics = {
            'psnr': self.to_running_device(PSNR()),
            'ssim': self.to_running_device(SSIM())
        }

    def init_training(self, cfg: Dict) -> None:
        super().init_training(cfg)

        self.repeat_num = cfg['TRAIN'].get('REPEAT_NUM', 1)
        self.init_meters()

        if hasattr(cfg, 'VAL'):
            self.init_validation(cfg)

    def init_validation(self, cfg) -> None:
        super().init_validation(cfg)

        self.val_table_list = cfg['VAL']['TABLE']['LIST']
        self.val_table_items = cfg['VAL']['TABLE']['ITEMS']

        self.val_data_loader = self.build_val_data_loader(cfg)

    def init_meters(self) -> None:
        self.register_epoch_meter('train/psnr', 'train', '{:.2f} (dB)')
        self.register_epoch_meter('val/psnr', 'val', '{:.2f} (dB)')

    @staticmethod
    def define_model(cfg: Dict) -> nn.Module:
        return build_model(cfg['MODEL']['NAME'], cfg['MODEL'].get('PARAM', {}))

    def build_train_data_loader(self, cfg: Dict):
        pass

    @staticmethod
    def build_train_dataset(cfg: Dict) -> Dataset:
        """Not need to be implemented
        """

        pass

    @staticmethod
    def build_val_dataset(cfg: Dict) -> Dataset:
        pass

    @staticmethod
    def build_val_data_loader(cfg: Dict) -> DataLoader:
        pass

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> None:
        raise NotImplementedError()

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]) -> None:
        raise NotImplementedError()

    def validate_large_dataset(self) -> None:
        raise NotImplementedError()