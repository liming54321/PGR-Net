# 1. 介绍

Image Shadow Removal

# 2. 环境

## 2.1. 使用服务器环境

### 2.1.1. Python

Python: 3.9.12

```shell
conda create -n istd python=3.9.12
```

### 2.1.2. 安装依赖


```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

# 3. 开始使用

## 3.1. 初始化代码


* 初始化checkpoints和results存储路径（xxx为名字）

```shell
# 服务器环境
mkdir -p /home/user/ckpt_save_path/checkpoints
mkdir -p /home/user/result_save_path/results
ln -s /home/user/ckpt_save_path/checkpoints checkpoints
ln -s /home/user/result_save_path/results results

```

## 3.2. 训练

```
easytrain -c configs/cfg_DGUNet_shadowFormer.py --gpus 0,1,2
```

## 3.4. 推理测试集图像生成结果图

每次修改`infer_ISTD_image.sh`中config和存储目录，设置推理使用的GPU：

```shell

CONFIG= configs/cfg_DGUNet_shadowFormer.py
OUTPUT_NAME=  DGUNet_shadowFormer  # 结果文件夹名字（不是完整路径，只填文件夹名字即可）
CKPT_NAME= # 权重文件路径
INPUT_NAME=Datasets/ISTD/test/

GPUS=0
```

推理：

```shell
bash infer_test_image.sh
```

## 3.6. Tensorboard

```shell
tensorboard --host 0.0.0.0 --logdir checkpoints
```

## 3.7. 模型信息（统计计算量、参数量）

```shell
python scripts/model_info_dw.py --human -c configs/cfg_DGUNet_shadowFormer.py -i "1,3,256,256" "1,1,256,256"
```

## 3.8. 性能评估（RMSE,SSIM,PSNR）

Matlab 2016

evaluate2.m

mask路径，结果图路径，目标图路径
