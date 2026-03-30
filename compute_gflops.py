import torch
from thop import profile, clever_format
import logging
import math
import os
import time

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from skimage import io

from models.model import Baseline
from train import train, test, visualize_testloader
from utils import ISPRS_dataset, convert_to_color, fix_random_seed, WHUDataset, YESegDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: 1) 按你项目改：导入并构建模型
    # from models.fd... import FDMFNet
    # model = FDMFNet(...)
    model_cfg = cfg.model
    model = Baseline(cfg=model_cfg, num_classes=6, in_chans=[3, 1])
    model = model.to(device)
    model.eval()

    # TODO: 2) 按你项目改：输入 shape
    # 例：分类常用 224，分割常用 512/1024
    rgb = torch.randn(1, 3, 256, 256).to(device)
    modal = torch.randn(1, 1, 256, 256).to(device)  # 第二模态 1 通道
    macs, params = profile(model, inputs=(rgb, modal), verbose=False)

    flops = 2 * macs
    gflops = flops / 1e9

    macs_f, params_f = clever_format([macs, params], "%.3f")
    print(f"MACs:   {macs_f}")
    print(f"Params: {params_f}")
    print(f"GFLOPs (FLOPs=2*MACs): {gflops:.3f}")

if __name__ == "__main__":
    main()