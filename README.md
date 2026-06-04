# PGR-Net

Official implementation of **Progressive Global-to-Local Image Restoration for Structure-Preserving Shadow Removal**.

This repository provides the implementation, configuration files, inference scripts, evaluation code, and pretrained models for reproducing the main experimental results reported in our paper.

---

## 1. Environment

### 1.1 Create Conda Environment

```bash
conda create -n PGR python=3.9.12
conda activate PGR
```

### 1.2 Install PyTorch 1.10.0 + CUDA 11.1

```bash
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

---

## 2. Dataset Preparation

We use the **ISTD** and **SRD** datasets for shadow removal evaluation. In our experiments, all images are resized to **256 × 256**.

Please download the datasets from the following links:

* ISTD dataset: [Download link](PUT_ISTD_DATASET_LINK_HERE)
* SRD dataset: [Download link](PUT_SRD_DATASET_LINK_HERE)

Please organize the ISTD dataset as follows:

```text
Datasets/
└── ISTD/
    ├── train/
    │   ├── input/
    │   ├── target/
    │   └── mask/
    └── test/
        ├── input/
        ├── target/
        └── mask/
```

Please organize the SRD dataset as follows:

```text
Datasets/
└── SRD/
    ├── train/
    │   ├── input/
    │   ├── target/
    │   └── mask/
    └── test/
        ├── input/
        ├── target/
        └── mask/
```

For SRD, the original dataset provides paired shadow and shadow-free images but does not provide official manually annotated shadow masks. Following the baseline setting, we use the **DHAN-generated SRD masks** adopted in the baseline implementation. The masks should be placed in the corresponding `mask/` folders and should have filenames consistent with the input images.

---

## 3. Training

To train the model, run:

```bash
easytrain -c configs/cfg_DGUNet_shadowFormer.py --gpus 0,1
```

Before training, please modify the dataset paths, checkpoint path, batch size, and GPU settings in the configuration file according to your environment.

---

## 4. Pretrained Models

The pretrained models can be downloaded from the following links:

| Dataset | Pretrained Model                                    |
| :-----: | :-------------------------------------------------- |
|   ISTD  | [Google Drive](PUT_ISTD_PRETRAINED_MODEL_LINK_HERE) |
|   SRD   | [Google Drive](PUT_SRD_PRETRAINED_MODEL_LINK_HERE)  |

Please place the downloaded pretrained models under:

```text
checkpoints/
├── pgrnet_istd.pth
└── pgrnet_srd.pth
```

---

## 5. Inference

Before inference, please modify the following settings in `infer_ISTD_image.sh`:

```bash
CONFIG=configs/cfg_DGUNet_shadowFormer.py
OUTPUT_NAME=test_name
CKPT_NAME="checkpoints/pgrnet_istd.pth"
INPUT_NAME=Datasets/ISTD/test/
GPUS=0
```

Then run:

```bash
bash infer_ISTD_image.sh
```

The restored images will be saved to the output directory specified by `OUTPUT_NAME`.

---

## 6. Evaluation

We provide `evaluate2.m` to compute the quantitative metrics, including **RMSE**, **SSIM**, and **PSNR**.

The evaluation script was tested with **MATLAB R2016**. Before running the script, please set the paths of the predicted results, ground-truth images, and shadow masks in `evaluate2.m`.

For example:

```matlab
% Path to the predicted shadow-free images
result_path = 'path/to/results';

% Path to the ground-truth shadow-free images
target_path = 'path/to/target';

% Path to the shadow masks
mask_path = 'path/to/mask';
```

Then run `evaluate2.m` in MATLAB to obtain the evaluation results.

## 7. Acknowledgements

This project uses the ISTD and SRD datasets. The SRD masks used in this repository follow the DHAN-generated masks adopted in the baseline implementation. We sincerely thank the authors of the related datasets and baseline methods for making their resources available to the research community.
