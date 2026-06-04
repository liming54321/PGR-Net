import os
import cv2
from einops import rearrange
from easytorch.config import Config
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union, Callable, Optional

from easytorch import AvgMeter
from easytorch.core.data_loader import build_data_loader, build_data_loader_ddp

from ..models import MODEL_REGISTRY
from .base_runner import BaseImageRunner
from ..utils.losses import LOSS_REGISTRY, calculate_loss, balance_loss
from ..utils.data_loaders.ISTD_dataloader import ISTDDataLoaderTrain, ISTDDataLoaderVal, RandomDataLoader


def forward_ISTD_image(runner: BaseImageRunner, noisy_image: torch.Tensor,
                       gt_image: Optional[torch.Tensor] = None, frame_callback: Optional[Callable] = None) -> None:
    """Forward video.

    Args:
        noisy_image (torch.Tensor): Noisy raw video data, shape: [N, 3, H, W]
        gt_image (Optional[torch.Tensor], optional): Ground truth raw video data, shape: [N, 3, H, W].
            Defaults to None.
        frame_callback (Optional[Callable], optional): Callback function in frame processing.
            Defaults to None.
            The function is called as ``frame_callback(i, noisy_frame, output_frame, gt_frame)``
    """

    scale = 2 ** runner.video_bits

    # forward
    output_frame_list = runner.model(noisy_image)

    if frame_callback is not None:
        noisy_frame = noisy_image
        gt_frame = None if gt_image is None else gt_image
        for i, output_frame in enumerate(output_frame_list):
            output_frame_list[i] = output_frame.clamp(0., (scale - 1) / scale)
        frame_callback(0, noisy_frame, output_frame_list, gt_frame)


class BaseISTDRunner(BaseImageRunner, metaclass=ABCMeta):
    def build_train_data_loader(self, cfg: Dict) -> None:
        dataset = ISTDDataLoaderTrain(cfg['TRAIN']['DATA']['DATASETS']['PARAM']['rgb_dir'],
                                      cfg['TRAIN']['DATA']['DATASETS']['PARAM']['img_options'])

        # dataset = RandomDataLoader(cfg['TRAIN']['DATA']['DATASETS']['PARAM']['rgb_dir'],
        #                            cfg['TRAIN']['DATA']['DATASETS']['PARAM']['img_options'])

        self.logger.info(
            'Dataset {}<{:d}> initialized'.format('HdR-HDM', len(dataset))
        )

        if torch.distributed.is_initialized():
            data_loader = build_data_loader_ddp(dataset, cfg['TRAIN']['DATA'])
        else:
            data_loader = build_data_loader(dataset, cfg['TRAIN']['DATA'])

        return data_loader

    def build_val_data_loader(self, cfg: Dict) -> None:
        dataset = ISTDDataLoaderVal(cfg['VAL']['DATA']['DATASETS']['PARAM']['rgb_dir'],
                                    cfg['VAL']['DATA']['DATASETS']['PARAM']['img_options'])
        # dataset = RandomDataLoader(cfg['TRAIN']['DATA']['DATASETS']['PARAM']['rgb_dir'],
        #                            cfg['TRAIN']['DATA']['DATASETS']['PARAM']['img_options'])
        return build_data_loader(dataset, cfg['VAL']['DATA'])

    def build_loss(self, loss_cfg: Dict) -> None:
        self.register_epoch_meter('train/total_loss', 'train', '{:.5f}')
        if hasattr(self.cfg, 'VAL'):
            self.register_epoch_meter('val/total_loss', 'val', '{:.5f}')

        self.loss_dict = {}
        for name, value in loss_cfg.items():
            loss = LOSS_REGISTRY.build(name, value.get('PARAM', {}))
            loss = self.to_running_device(loss)

            self.loss_dict[name] = {
                'loss': loss,
                'weight': value.get('WEIGHT', 1.0),
                'balance': value.get('BALANCE', False)
            }
            self.register_epoch_meter('train/' + name.lower(), 'train', '{:.5f}')
            if hasattr(self.cfg, 'VAL'):
                self.register_epoch_meter('val/' + name.lower(), 'val', '{:.5f}')

        self.logger.info('Init loss: {}'.format(str(self.loss_dict)))

    def compute_loss(self, y_pred: torch.Tensor, y_gt: torch.Tensor,
                     update_meter: bool = True, training: bool = True, **kwargs) -> torch.Tensor:
        meter_prefix = 'train/' if training else 'val/'

        total_loss = 0.0
        balanced_loss_list = []

        for name, value in self.loss_dict.items():
            loss = calculate_loss(value['loss'], y_pred, y_gt, **kwargs)
            weighted_loss = loss * value['weight']
            if value['balance']:
                balanced_loss_list.append(weighted_loss)
            else:
                total_loss += weighted_loss
            if update_meter:
                self.update_epoch_meter(meter_prefix + name.lower(), loss.item())

        # balance loss
        if len(balanced_loss_list) != 0:
            total_loss += balance_loss(balanced_loss_list)

        if update_meter:
            self.update_epoch_meter(meter_prefix + 'total_loss', total_loss.item())
        return total_loss

    def preprocess_data_device(self, data: Tuple[torch.Tensor]) -> torch.Tensor:
        tar_img, inp_img, filename = data
        tar_img = self.to_running_device(tar_img)
        inp_img = self.to_running_device(inp_img)
        # filename = self.to_running_device(filename)
        return tar_img, inp_img, filename

    def data_augmentation(self, cfg: Dict, images_0, images_1):
        batchsize, channels, w, h = images_0.size()

        # Random resize
        if hasattr(cfg['TRAIN']['DATA']['DATASETS']['AUG'], 'RESIZE'):
            l_limit = cfg['TRAIN']['DATA']['DATASETS']['AUG']['RESIZE'][0]
            u_limit = cfg['TRAIN']['DATA']['DATASETS']['AUG']['RESIZE'][1]
            # factor = torch.FloatTensor().uniform_(l_limit, u_limit)
            factor = np.random.uniform(l_limit, u_limit)
            new_w = int(factor * w)
            new_h = int(factor * h)
            resized_images = torch.zeros([batchsize, channels, int(new_w), int(new_h)])
            for i in range(batchsize):
                resized_images[i] = F.interpolate(images_0[i].unsqueeze(0), size=(new_w, new_h), mode='bilinear')
            images_0 = resized_images
            for i in range(batchsize):
                resized_images[i] = F.interpolate(images_1[i].unsqueeze(0), size=(new_w, new_h), mode='bilinear')
            images_1 = resized_images

        # PhotoMetricDistortion
        # if hasattr(cfg['TRAIN']['DATA']['DATASETS']['AUG'] , 'PHOTOMETRIC'):
        #     augmented_images_0 = torch.zeros_like(images_0)
        #     augmented_images_1 = torch.zeros_like(images_0)
        #     for i in range(batchsize):
        #         for c in range(channels):
        #             distorted_image = images_0[i, c]
        #             alpha = torch.FloatTensor(3).uniform_(0.8, 1.2)
        #             distorted_image *= alpha.view(3, 1, 1)
        #             distorted_image += torch.FloatTensor(3).uniform_(-0.2, 0.2).view(3, 1, 1)
        #             distorted_image = torch.clamp(distorted_image, 0, 1)
        #             augmented_images_0[i, c] = distorted_image

        #             distorted_image = images_1[i, c]
        #             distorted_image *= alpha.view(3, 1, 1)
        #             distorted_image += torch.FloatTensor(3).uniform_(-0.2, 0.2).view(3, 1, 1)
        #             distorted_image = torch.clamp(distorted_image, 0, 1)
        #             augmented_images_1[i, c] = distorted_image
        #     images_0 = augmented_images_0
        #     images_1 = augmented_images_1

        # Random flip
        if hasattr(cfg['TRAIN']['DATA']['DATASETS']['AUG'], 'FLIP'):
            flip_prob = torch.FloatTensor(batchsize).uniform_(0, 1)
            for i in range(batchsize):
                for c in range(channels):
                    if flip_prob[i] < cfg['TRAIN']['DATA']['DATASETS']['AUG']['FLIP']:
                        images_0[i, c] = images_0[i, c].flip(1)
                        images_1[i, c] = images_1[i, c].flip(1)

        # Normalize
        normalized_images_0 = images_0 / 255.0
        normalized_images_1 = images_1 / 255.0

        # Pad to multiple of 32
        padded_images_0 = F.pad(normalized_images_0, (0, 0, 0, (32 - new_w.max() % 32) % 32))
        padded_images_1 = F.pad(normalized_images_1, (0, 0, 0, (32 - new_w.max() % 32) % 32))

        return padded_images_0, padded_images_1

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> None:
        """
        training iter, train images

        @param epoch: epoch
        @param iter_index: iter index
        @param data: data yield by dataloader
            shape: [N, 3, H, W]
            dtype: uint8
        @return: None
        """

        # pylint: disable=unused-argument
        def frame_callback(i: int, noisy_frame: torch.Tensor, output_frame: torch.Tensor,
                           gt_frame: Optional[torch.Tensor] = None) -> None:
            # backward & update parameters
            loss = sum([self.compute_loss(output_frame[i], gt_frame) for i in range(len(output_frame))])
            # loss = self.compute_loss(output_frame, gt_frame)
            self.backward(loss)

            # calculate PSNR
            self.update_epoch_meter('train/psnr', self.metrics['psnr'](output_frame[0], gt_frame).item())

        for _ in range(self.repeat_num):
            # degrade video
            # TODO
            gt_image, noisy_image, _ = self.preprocess_data_device(data)
            # gt_image, noisy_image = self.data_augmentation(self.cfg, gt_image, noisy_image)

            self.forward_image(noisy_image, gt_image, frame_callback)

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]) -> None:
        val_meters = OrderedDict(tuple(map(lambda item: (item[0], AvgMeter()), self.val_table_items)))

        # pylint: disable=unused-argument
        def frame_callback(output_frame: torch.Tensor, gt_frame: torch.Tensor = None) -> None:
            if val_meters.get('LOSS') is not None:
                value = self.compute_loss(output_frame[0], gt_frame, update_meter=False).item()
                val_meters['LOSS'].update(value)
            if val_meters.get('PSNR') is not None:
                value = self.metrics['psnr'](output_frame[0], gt_frame).item()
                val_meters['PSNR'].update(value)

            # calculate PSNR
            self.update_epoch_meter('val/psnr', self.metrics['psnr'](output_frame[0], gt_frame).item())

        gt_image, noisy_image, _ = self.preprocess_data_device(data)
        self.validate_frame(noisy_image, gt_image, frame_callback)

        return [meter.avg for meter in val_meters.values()]

    def forward_image(self, noisy_image: torch.Tensor, gt_image: Optional[torch.Tensor] = None,
                      frame_callback: Optional[Callable] = None) -> None:
        forward_ISTD_image(self, noisy_image, gt_image, frame_callback)

    def validate_frame(self, noisy_image: torch.Tensor, gt_image: Optional[torch.Tensor] = None,
                       frame_callback: Optional[Callable] = None) -> None:
        scale = 2 ** self.video_bits

        output_frame_list = self.model(noisy_image)
        for i, output_frame in enumerate(output_frame_list):
            output_frame_list[i] = output_frame.clamp(0., (scale - 1) / scale)
        frame_callback(output_frame_list, gt_image)
        # print(output_frame_list)

    # TODO: flexible validation tables
    @torch.no_grad()
    def validate_dataset(self) -> None:
        self.logger.info('Running validation on Large Val Dataset...')
        self.model.eval()
        self.register_epoch_meter('val/psnr', 'val', '{:.2f} (dB)')

        loss = 0
        psnr = 0
        index = 0
        for iter_index, data in enumerate(self.val_data_loader):
            self.logger.info('Validating image {:d}...'.format(iter_index))
            result = self.val_iters(iter_index, data)

            loss += result[0]
            psnr += result[1]
            index += 1

        self.logger.info('Loss: {}; PSNR: {}...'.format(loss / index, psnr / index))

    @torch.no_grad()
    def infer_istd(self, infer_set, results_save_dir: str, max_frame_num: int) -> None:
        self.logger.info('Save video to {}'.format(results_save_dir))
        # make result save dir
        if not os.path.isdir(results_save_dir):
            os.makedirs(results_save_dir)

        frame_num = len(infer_set) if max_frame_num is None else min(max_frame_num, len(infer_set))

        print(len(infer_set))
        # print(frame_num)
        raw_names = []
        for _ in range(len(infer_set)):
            raw_names.append(infer_set[_][-1])
        print('finish')

        output_video = self.infer_istd_frame(infer_set, frame_num)
        # print('output video shape is {}'.format(output_video.shape)) # shape is [540, 64, 64, 3], 64 deponds on the img_options

        for i in range(frame_num):
            current_frame = output_video[i, ...]
            raw_name = raw_names[i] + '.png'
            # file_path = os.path.join(results_save_dir, '{:05d}.png'.format(i))
            file_path = os.path.join(results_save_dir, raw_name)
            print(file_path)
            print('processing frame {}, shape is: {}, save directory is {}'.format(i, current_frame.shape, file_path))
            current_frame = np.array(current_frame)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, current_frame)

    def infer_istd_frame(self, infer_set, frame_num) -> torch.Tensor:
        output_frames_list = []

        # pylint: disable=unused-argument
        def frame_callback(output_frame: torch.Tensor, gt_frame: torch.Tensor = None, ldr_s: torch.Tensor = None,
                           ldr_l: torch.Tensor = None, ) -> None:
            output_frame = output_frame[0].cpu().detach()
            output_frame = rearrange(output_frame, 'b c h w -> b h w c')
            # print('output frame shape is {}'.format(output_frame.shape)) # shape is [1, 64, 64, 3]

            # unpack raw
            output_frame_scale = (output_frame * (2 ** self.video_bits)).round()
            output_frames_list.append(output_frame_scale.cpu())

        for i in range(frame_num):
            gt, input, _ = infer_set[i]
            gt = self.to_running_device(rearrange(gt, '(b c) h w -> b c h w', b=1))
            # print(gt.shape)
            input = self.to_running_device(rearrange(input, '(b c) h w -> b c h w', b=1))
            # print(input.shape)

            self.validate_frame(input, gt, frame_callback=frame_callback)

        return torch.cat(output_frames_list, dim=0) if len(output_frames_list) != 0 else None

    def on_validating_end(self, train_epoch: Optional[int]):
        # `None` means validation mode
        if train_epoch is not None:
            self.save_best_model(train_epoch, 'val/psnr', greater_best=True)


