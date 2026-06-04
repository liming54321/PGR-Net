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
                       mask_image: torch.Tensor, gt_image: Optional[torch.Tensor] = None,
                       frame_callback: Optional[Callable] = None) -> None:
    scale = 2 ** runner.video_bits
    output_frame_list = runner.model(noisy_image, mask_image)
    if frame_callback is not None:

        for i in range(len(output_frame_list)):
            output_frame_list[i] = output_frame_list[i].clamp(0., (scale - 1) / scale)

        import inspect
        param_count = len(inspect.signature(frame_callback).parameters)
        if param_count == 2:
            frame_callback(output_frame_list, gt_image)
        elif param_count == 5:
            frame_callback(0, noisy_image, mask_image, output_frame_list, gt_image)
        else:
            raise RuntimeError(f"Unsupported frame_callback with {param_count} parameters.")

class BaseISTDRunner(BaseImageRunner, metaclass=ABCMeta):
    def build_train_data_loader(self, cfg: Dict) -> None:
        dataset = ISTDDataLoaderTrain(cfg['TRAIN']['DATA']['DATASETS']['PARAM']['rgb_dir'],
                                      cfg['TRAIN']['DATA']['DATASETS']['PARAM']['img_options'])
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
        if len(balanced_loss_list) != 0:
            total_loss += balance_loss(balanced_loss_list)
        if update_meter:
            self.update_epoch_meter(meter_prefix + 'total_loss', total_loss.item())
        return total_loss

    def preprocess_data_device(self, data: Tuple[torch.Tensor]) -> torch.Tensor:
        tar_img, inp_img, mask_img, filename = data
        tar_img = self.to_running_device(tar_img)
        inp_img = self.to_running_device(inp_img)
        mask_img = self.to_running_device(mask_img)
        return tar_img, inp_img, mask_img, filename

    def data_augmentation(self, cfg: Dict, images_0, images_1):
        batchsize, channels, w, h = images_0.size()
        if hasattr(cfg['TRAIN']['DATA']['DATASETS']['AUG'], 'RESIZE'):
            l_limit = cfg['TRAIN']['DATA']['DATASETS']['AUG']['RESIZE'][0]
            u_limit = cfg['TRAIN']['DATA']['DATASETS']['AUG']['RESIZE'][1]
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
        if hasattr(cfg['TRAIN']['DATA']['DATASETS']['AUG'], 'FLIP'):
            flip_prob = torch.FloatTensor(batchsize).uniform_(0, 1)
            for i in range(batchsize):
                for c in range(channels):
                    if flip_prob[i] < cfg['TRAIN']['DATA']['DATASETS']['AUG']['FLIP']:
                        images_0[i, c] = images_0[i, c].flip(1)
                        images_1[i, c] = images_1[i, c].flip(1)
        normalized_images_0 = images_0 / 255.0
        normalized_images_1 = images_1 / 255.0
        padded_images_0 = F.pad(normalized_images_0, (0, 0, 0, (32 - new_w.max() % 32) % 32))
        padded_images_1 = F.pad(normalized_images_1, (0, 0, 0, (32 - new_w.max() % 32) % 32))

        return padded_images_0, padded_images_1

    def train_iters(self, epoch: int, iter_index: int, data):
        import torch
        import torch.nn.functional as F
        from src.utils.losses.other_losses import GradientLoss
        grad_loss = GradientLoss()
        if epoch <= 5:
            edge_loss_weight = 0.0
        elif 6 <= epoch <= 15:
            edge_loss_weight = (epoch - 5) / 10 * 0.03
        elif 16 <= epoch <= 100:
            edge_loss_weight = 0.03
        else:
            edge_loss_weight = min(0.06, 0.03 + (epoch - 100) / 200 * 0.03)
        if epoch <= 50:
            nonshadow_weight = 0.2
        elif 51 <= epoch <= 200:
            nonshadow_weight = 0.2 + (epoch - 50) / 150 * 0.8
        else:
            nonshadow_weight = 1.0
        def _normalize_mask(m: torch.Tensor) -> torch.Tensor:
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
            elif m.ndim == 3:
                m = m.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
            m = m.float()
            if m.shape[1] != 1: m = m[:, :1, :, :]
            if m.max().item() > 1.0: m = m / 255.0
            return m.clamp(0, 1)
        RING_K = 3
        RING_W = 0.25
        def boundary_ring(mm: torch.Tensor, k: int = RING_K) -> torch.Tensor:
            dil = F.max_pool2d(mm, kernel_size=k, stride=1, padding=k // 2)
            ero = 1 - F.max_pool2d(1 - mm, kernel_size=k, stride=1, padding=k // 2)
            return (dil - ero).clamp(0, 1)

        def frame_callback(i, noisy_frame, mask_frame, output_frame, gt_frame=None):
            if isinstance(output_frame, torch.Tensor):
                output_frame = [output_frame]
            w_stage = [1.0, 0.5, 0.5, 1.0] if len(output_frame) == 4 else [1.0] * len(output_frame)
            m = _normalize_mask(mask_frame)  # [B,1,H,W]
            ring = boundary_ring(m, k=RING_K)
            eps = 1e-6
            occ_s = m.mean() + eps
            occ_ns = (1 - m).mean() + eps
            occ_r = ring.mean() + eps
            total_loss = 0.0
            for j, pred in enumerate(output_frame):
                if pred.ndim == 3: pred = pred.unsqueeze(0)
                target = gt_frame if not (
                            isinstance(gt_frame, (list, tuple)) and len(gt_frame) == len(output_frame)) else gt_frame[j]
                if target is not None and target.ndim == 3: target = target.unsqueeze(0)
                loss_main = self.compute_loss(pred, target)
                shadow_boost = 1.15 if epoch <= 150 else 1.10
                alpha_ns = min(nonshadow_weight, 0.8)
                ring_w = 0.35 if epoch <= 60 else (0.30 if epoch <= 100 else 0.25)
                Lg_s = grad_loss(pred * m, target * m) / occ_s
                Lg_ns = grad_loss(pred * (1 - m), target * (1 - m)) / occ_ns
                Lg_r = grad_loss(pred * ring, target * ring) / occ_r
                loss_grad = shadow_boost * Lg_s + alpha_ns * Lg_ns + ring_w * Lg_r
                if epoch <= 60:
                    warm = 0.2 * (1.0 - epoch / 60.0)  # 0.2 -> 0.0
                    loss_main = loss_main + warm * (self.compute_loss(pred * m, target * m) / (occ_s))

                w = w_stage[j]
                total_loss += w * (loss_main + edge_loss_weight * loss_grad)
                if j == 0:
                    self.update_epoch_meter('train/edge_loss', loss_grad.item())
                    self.update_epoch_meter('train/psnr', self.metrics['psnr'](pred, target).item())

            self.backward(total_loss)
        for _ in range(self.repeat_num):
            gt_image, noisy_image, mask_image, _ = self.preprocess_data_device(data)
            self.forward_image(noisy_image, mask_image, gt_image, frame_callback)

    def val_iters(self, iter_index: int, data):
        import torch
        import torch.nn.functional as F
        from collections import OrderedDict
        from src.utils.losses.other_losses import GradientLoss

        grad_loss = GradientLoss()
        val_meters = OrderedDict(tuple(map(lambda item: (item[0], AvgMeter()), self.val_table_items)))

        epoch = int(getattr(self, 'current_epoch', 0))
        if epoch <= 50:
            nonshadow_weight = 0.2
        elif 51 <= epoch <= 200:
            nonshadow_weight = 0.2 + (epoch - 50) / 150 * 0.8
        else:
            nonshadow_weight = 1.0
        if epoch <= 5:
            edge_loss_weight = 0.0
        elif 6 <= epoch <= 15:
            edge_loss_weight = (epoch - 5) / 10 * 0.03
        elif 16 <= epoch <= 100:
            edge_loss_weight = 0.03
        else:
            edge_loss_weight = min(0.06, 0.03 + (epoch - 100) / 200 * 0.03)
        RING_K = 3
        def _normalize_mask(m: torch.Tensor) -> torch.Tensor:
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(1)
            m = m.float()
            if m.shape[1] != 1: m = m[:, :1, :, :]
            if m.max().item() > 1.0: m = m / 255.0
            return m.clamp(0, 1)
        def boundary_ring(mm: torch.Tensor, k=RING_K):
            dil = F.max_pool2d(mm, kernel_size=k, stride=1, padding=k // 2)
            ero = 1 - F.max_pool2d(1 - mm, kernel_size=k, stride=1, padding=k // 2)
            return (dil - ero).clamp(0, 1)

        FINAL_IDX = 0
        loss_dict = getattr(self, 'loss_dict', {}) or {}
        perceptual_fns, charbonnier_fns = [], []
        for name, cfg in loss_dict.items():
            lname = str(name).lower()
            fn = cfg.get('loss', None)
            if callable(fn):
                if ('percep' in lname) or ('vgg' in lname) or ('lpips' in lname) or ('style' in lname):
                    perceptual_fns.append(fn)
                if 'charbon' in lname:
                    charbonnier_fns.append(fn)
        def frame_callback(output_frame, gt_frame=None, mask_frame=None, noisy_frame=None):
            if isinstance(output_frame, torch.Tensor):
                output_frame = [output_frame]
            pred = output_frame[FINAL_IDX]
            if pred.ndim == 3: pred = pred.unsqueeze(0)
            gt = gt_frame.unsqueeze(0) if (gt_frame is not None and gt_frame.ndim == 3) else gt_frame
            if gt is None:
                return
            total = self.compute_loss(pred, gt, update_meter=False, training=False).item()
            if val_meters.get('val/total_loss') is not None:
                val_meters['val/total_loss'].update(total)

            if (mask_frame is not None) and (val_meters.get('val/edge_loss') is not None):
                m = _normalize_mask(mask_frame)
                ring = boundary_ring(m, k=RING_K)
                eps = 1e-6
                occ_s = m.mean() + eps
                occ_ns = (1 - m).mean() + eps
                occ_r = ring.mean() + eps
                shadow_boost = 1.15 if epoch <= 150 else 1.10
                alpha_ns = min(nonshadow_weight, 0.8)
                ring_w = 0.35 if epoch <= 60 else (0.30 if epoch <= 100 else 0.25)
                Lg_s = grad_loss(pred * m, gt * m) / occ_s
                Lg_ns = grad_loss(pred * (1 - m), gt * (1 - m)) / occ_ns
                Lg_r = grad_loss(pred * ring, gt * ring) / occ_r
                edge_core = shadow_boost * Lg_s + alpha_ns * Lg_ns + ring_w * Lg_r
                edge_val = (edge_loss_weight * edge_core).item()
                val_meters['val/edge_loss'].update(edge_val)
            if (val_meters.get('val/perceptualloss') is not None) and (len(perceptual_fns) > 0):
                vals = []
                for fn in perceptual_fns:
                    v = fn(pred, gt)
                    v = v.mean().item() if isinstance(v, torch.Tensor) else float(v)
                    vals.append(v)
                val_meters['val/perceptualloss'].update(sum(vals) / len(vals))
            if (val_meters.get('val/charbonnierloss') is not None) and (len(charbonnier_fns) > 0):
                vals = []
                for fn in charbonnier_fns:
                    v = fn(pred, gt)
                    v = v.mean().item() if isinstance(v, torch.Tensor) else float(v)
                    vals.append(v)
                val_meters['val/charbonnierloss'].update(sum(vals) / len(vals))
            psnr_val = float(self.metrics['psnr'](pred, gt).item())
            if val_meters.get('val/psnr') is not None:
                val_meters['val/psnr'].update(psnr_val)
            self.update_epoch_meter('val/psnr', psnr_val)
        gt_image, noisy_image, mask_image, _ = self.preprocess_data_device(data)
        self.validate_frame(noisy_image, mask_image, gt_image, frame_callback)
        return [meter.avg for meter in val_meters.values()]

    def forward_image(self,
                      noisy_image: torch.Tensor,
                      mask_image: torch.Tensor,
                      gt_image: Optional[torch.Tensor] = None,
                      frame_callback: Optional[Callable] = None) -> None:
        forward_ISTD_image(self, noisy_image, mask_image, gt_image, frame_callback)
    def validate_frame(self, noisy_image, mask_image, gt_image=None, frame_callback=None):
        import torch
        self.model.eval()
        scale = 2 ** self.video_bits
        def _aug(x, mode):
            if mode == 0:  return x
            if mode == 1:  return x.flip(-1)
            if mode == 2:  return x.flip(-2)
            if mode == 3:  return x.flip(-1).flip(-2)
            if mode == 4:  return x.rot90(1, dims=[-2, -1])
            if mode == 5:  return x.rot90(1, dims=[-2, -1]).flip(-1)
            if mode == 6:  return x.rot90(2, dims=[-2, -1])
            if mode == 7:  return x.rot90(3, dims=[-2, -1])
            return x
        def _deaug(x, mode):
            if mode == 0:  return x
            if mode == 1:  return x.flip(-1)
            if mode == 2:  return x.flip(-2)
            if mode == 3:  return x.flip(-2).flip(-1)
            if mode == 4:  return x.rot90(3, dims=[-2, -1])
            if mode == 5:  return x.flip(-1).rot90(3, dims=[-2, -1])
            if mode == 6:  return x.rot90(2, dims=[-2, -1])
            if mode == 7:  return x.rot90(1, dims=[-2, -1])
            return x
        with torch.no_grad():
            outs_accum = None
            for mode in range(8):
                xi = _aug(noisy_image, mode)
                mi = _aug(mask_image, mode)
                outs = self.model(xi, mi)
                if isinstance(outs, torch.Tensor):
                    outs = [outs]
                outs = [_deaug(y, mode) for y in outs]
                if outs_accum is None:
                    outs_accum = [o.clone() for o in outs]
                else:
                    outs_accum = [a + b for a, b in zip(outs_accum, outs)]
            output_frame_list = [o / 8.0 for o in outs_accum]
            output_frame_list = [o.clamp(0., (scale - 1) / scale) for o in output_frame_list]
        if frame_callback is not None:
            frame_callback(output_frame_list, gt_image, mask_image, noisy_image)
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
        if not os.path.isdir(results_save_dir):
            os.makedirs(results_save_dir)
        frame_num = len(infer_set) if max_frame_num is None else min(max_frame_num, len(infer_set))
        print(len(infer_set))
        raw_names = []
        for _ in range(len(infer_set)):
            raw_names.append(infer_set[_][-1])
        print('finish')
        output_video = self.infer_istd_frame(infer_set, frame_num)
        for i in range(frame_num):
            current_frame = output_video[i, ...]
            raw_name = raw_names[i] + '.png'
            file_path = os.path.join(results_save_dir, raw_name)
            print(file_path)
            print('processing frame {}, shape is: {}, save directory is {}'.format(i, current_frame.shape, file_path))
            current_frame = np.array(current_frame)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, current_frame)

    def infer_istd_frame(self, infer_set, frame_num) -> torch.Tensor:
        output_frames_list = []
        def frame_callback(output_frame: torch.Tensor, gt_frame: torch.Tensor = None, ldr_s: torch.Tensor = None,
                           ldr_l: torch.Tensor = None, ) -> None:
            output_frame = output_frame[0].cpu().detach()
            output_frame = rearrange(output_frame, 'b c h w -> b h w c')
            output_frame_scale = (output_frame * (2 ** self.video_bits)).round()
            output_frames_list.append(output_frame_scale.cpu())

        for i in range(frame_num):
            gt, input, mask, _ = infer_set[i]
            gt = self.to_running_device(rearrange(gt, '(b c) h w -> b c h w', b=1))
            input = self.to_running_device(rearrange(input, '(b c) h w -> b c h w', b=1))
            mask = self.to_running_device(rearrange(mask, '(b c) h w -> b c h w', b=1))
            self.validate_frame(input, mask, gt, frame_callback=frame_callback)

        return torch.cat(output_frames_list, dim=0) if len(output_frames_list) != 0 else None

    def on_validating_end(self, train_epoch: Optional[int]):
        if train_epoch is not None:
            self.save_best_model(train_epoch, 'val/psnr', greater_best=True)