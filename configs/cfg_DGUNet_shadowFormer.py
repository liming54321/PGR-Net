import os

from easydict import EasyDict

from src.runners.ISTD_image_runner import BaseISTDRunner

CFG = EasyDict()

CFG.DESC = 'DGUNet_shadowformer model for image shadow removal'  #
CFG.RUNNER = BaseISTDRunner
CFG.GPU_NUM = 2

CFG.ENV = EasyDict()
CFG.ENV.SEED = 3407

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'DGUNet_shadowFormer'
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

CFG.OPTION = EasyDict()
CFG.OPTION.VIDEO_BITS = 8

CFG.TRAIN = EasyDict()

CFG.TRAIN.NUM_EPOCHS = 300
CFG.TRAIN.REPEAT_NUM = 1
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints', 'train_name',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = list(range(10, 200, 10)) + list(range(200, CFG.TRAIN.NUM_EPOCHS, 1))

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'Adam'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 1e-4,
    'betas': [0.9, 0.99],
    'eps': 1e-8
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'milestones': [120, 250],
    'gamma': 0.1
}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.TOTAL_BATCH_SIZE = 8
CFG.TRAIN.DATA.BATCH_SIZE = CFG.TRAIN.DATA.TOTAL_BATCH_SIZE // (CFG.GPU_NUM * CFG.get('DIST_NODE_NUM', 1))
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.DATA.PIN_MEMORY = False

CFG.TRAIN.DATA.DATASETS = EasyDict()
CFG.TRAIN.DATA.DATASETS.PARAM = {
    'rgb_dir': './Datasets/ISTD/train/shadow/',
    'img_options': {'patch_size': 256}
}
CFG.TRAIN.DATA.DATASETS.AUG = EasyDict()
CFG.TRAIN.DATA.DATASETS.AUG.RESIZE = (0.8, 1.2)
CFG.TRAIN.DATA.DATASETS.AUG.FLIP = 0.5

CFG.TRAIN.LOSS = EasyDict()
CFG.TRAIN.LOSS.EDGE_LOSS = EasyDict()
CFG.TRAIN.LOSS.EDGE_LOSS.WEIGHT = 1.05
CFG.TRAIN.LOSS.PerceptualLoss = EasyDict()
CFG.TRAIN.LOSS.PerceptualLoss.WEIGHT = 1.0

CFG.TRAIN.LOSS.CharbonnierLoss = EasyDict()
CFG.TRAIN.LOSS.CharbonnierLoss.WEIGHT = 1.0

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 10000
CFG.VAL.TABLE = EasyDict()

CFG.VAL.TABLE.LIST = [
    (None, None),
    (15.0, 512.0)
]
CFG.VAL.TABLE.ITEMS = [
    ('LOSS', '{:.4e}'),
]

CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PREFETCH = True
CFG.VAL.DATA.PIN_MEMORY = False

CFG.VAL.DATA.DATASETS = EasyDict()
CFG.VAL.DATA.DATASETS.PARAM = {
    'rgb_dir': './Datasets/ISTD/test/shadow/',
    'img_options': {'patch_size': 256}
}