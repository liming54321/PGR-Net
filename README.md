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

We use the **ISTD** and **SRD** datasets for shadow removal evaluation. The dataset files provided in the download links below have already been resized to **256 × 256**, which is the image resolution used in our main experiments.

The train/test splits used in this repository strictly follow the original benchmark settings adopted in the corresponding datasets and previous shadow removal studies. We do not create new random splits.

For **ISTD**, the dataset contains **1,330 training triplets** and **540 testing triplets**. Each triplet consists of a shadow image, a corresponding shadow-free image, and a shadow mask.

For **SRD**, the dataset contains **2,680 training pairs** and **408 testing pairs**. Each pair consists of a shadow image and its corresponding shadow-free image. Since the original SRD dataset does not provide official shadow masks, we directly use the **DHAN-provided predicted SRD masks**, i.e., the SRD mask files provided with the base mask-based shadow-removal implementation adopted in this work. According to the description of that implementation, the shadow masks for SRD were generated using the **DHAN** method. These masks have also been adopted in previous mask-based shadow-removal studies and are used as fixed mask priors in our experiments.

Please download the datasets from the following links:

* ISTD dataset: [Download link](https://drive.google.com/file/d/17_2AmU5ujm3uh-hNfSPJvOpVxFLcv87m/view?usp=drive_link)
* SRD dataset: [Download link](https://drive.google.com/file/d/1Sv6yKBQAh3LCGVn16Zy66asuhRI3gVhQ/view?usp=drive_link)

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

Please make sure that the file names in `input/`, `target/`, and `mask/` are matched one-to-one within each split.

---

## 3. Training

To train the model, run:

```bash
easytrain -c configs/cfg_DGUNet_shadowFormer.py --gpus 0,1
```

Before training, please modify the dataset paths, checkpoint path, batch size, and GPU settings in the configuration file according to your environment.

For ISTD training, please set the dataset path to:

```text
Datasets/ISTD/train/
```

For SRD training, please set the dataset path to:

```text
Datasets/SRD/train/
```

The same training command can be used for both datasets after modifying the corresponding dataset paths and output checkpoint settings in the configuration file.

---

## 4. Pretrained Models

The pretrained models can be downloaded from the following links:

| Dataset | Pretrained Model                                                                                      |
| :-----: | :---------------------------------------------------------------------------------------------------- |
|   ISTD  | [Google Drive](https://drive.google.com/file/d/1Dfk0YW505j4hqo37_feXjFQDYXSKcn0h/view?usp=drive_link) |
|   SRD   | [Google Drive](https://drive.google.com/file/d/1_KVokBCDRSEAUol85xR8gORHwxScoS39/view?usp=drive_link) |

Please place the downloaded pretrained models under:

```text
checkpoints/
├── pgrnet_istd.pth
└── pgrnet_srd.pth
```

---

## 5. Inference

The inference procedure is the same for ISTD and SRD. Before inference, please modify the dataset path, checkpoint path, output name, and GPU setting in the inference script according to the dataset being evaluated.

### 5.1 ISTD Inference

For ISTD, please modify the following settings in `infer_ISTD_image.sh`:

```bash
CONFIG=configs/cfg_DGUNet_shadowFormer.py
OUTPUT_NAME=test_istd
CKPT_NAME="checkpoints/pgrnet_istd.pth"
INPUT_NAME=Datasets/ISTD/test/
GPUS=0
```

Then run:

```bash
bash infer_ISTD_image.sh
```

The restored images will be saved to the output directory specified by `OUTPUT_NAME`.

### 5.2 SRD Inference

For SRD, the inference process is identical to ISTD. You can either modify `infer_ISTD_image.sh` directly or copy it as a separate SRD inference script. The key settings should be changed as follows:

```bash
CONFIG=configs/cfg_DGUNet_shadowFormer.py
OUTPUT_NAME=test_srd
CKPT_NAME="checkpoints/pgrnet_srd.pth"
INPUT_NAME=Datasets/SRD/test/
GPUS=0
```

Then run the inference script in the same way:

```bash
bash infer_ISTD_image.sh
```

If you copy the script as `infer_SRD_image.sh`, you can run:

```bash
bash infer_SRD_image.sh
```

---

## 6. Evaluation

We provide `evaluate2.m` to compute the quantitative metrics, including **RMSE**, **SSIM**, and **PSNR**.

Following the evaluation protocol used in the paper, the metrics are reported on three regions:

* **shadow region**
* **non-shadow region**
* **all image region**

The shadow and non-shadow regions are determined according to the corresponding shadow masks. The all image region is computed over the whole image.

The evaluation script was tested with **MATLAB R2016**. Before running the script, please set the paths of the predicted results, ground-truth images, and shadow masks in `evaluate2.m`.

```matlab
% Path to the shadow masks
maskdir = 'path/to/mask';

% Path to the restored result images
shadowdir = 'path/to/results';

% Path to the ground-truth shadow-free images
freedir = 'path/to/target';
```

Then run `evaluate2.m` in MATLAB to obtain the evaluation results.

For ISTD evaluation, the paths should correspond to:

```text
Datasets/ISTD/test/mask/
Datasets/ISTD/test/target/
```

For SRD evaluation, the paths should correspond to:

```text
Datasets/SRD/test/mask/
Datasets/SRD/test/target/
```

---

## 7. Reproducing Quantitative Results

To reproduce the quantitative results reported in the paper, please follow the same dataset split, image resolution, pretrained model, inference procedure, and evaluation protocol described above.

### 7.1 ISTD

1. Download the ISTD dataset and the ISTD pretrained model.
2. Organize the dataset as described in Section 2.
3. Run inference using `checkpoints/pgrnet_istd.pth`.
4. Run `evaluate2.m` using the ISTD test masks and ground-truth images.

### 7.2 SRD

1. Download the SRD dataset and the SRD pretrained model.
2. Organize the dataset as described in Section 2.
3. Run inference using `checkpoints/pgrnet_srd.pth`.
4. Run `evaluate2.m` using the SRD test masks and ground-truth images.

The comparison results of existing baseline methods reported in the paper are cited from their original papers unless otherwise specified. The results of PGR-Net are obtained using the pretrained models, dataset splits, and evaluation protocol provided in this repository.

---

## 8. Acknowledgements

This project uses the ISTD and SRD datasets. The SRD masks used in this repository are taken from the base implementation on which our code is built, where the SRD shadow masks were generated using the DHAN method. We sincerely thank the authors of the related datasets and baseline methods for making their resources available to the research community.

---

## 9. Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{li2026progressive,
  title={Progressive Global-to-Local Image Restoration for Structure-Preserving Shadow Removal},
  author={Li, Ming and Hu, Weijian and Cao, Yali and Li, Lingfang and Zhang, Jikai},
  journal={The Visual Computer},
  year={2026},
  note={Under review}
}
```
