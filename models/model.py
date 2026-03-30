import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch_dct as DCT

import matplotlib.pyplot as plt
import numpy as np
import torch_dct as DCT
import os

import torch
import torch_dct as DCT
import matplotlib.pyplot as plt
import os

def denormalize_dsm(x):
    """
    DSM单通道数据的反归一化（根据你的实际归一化方式调整）
    若DSM未归一化，直接返回x即可
    """
    # 示例：如果DSM归一化到[0,1]，无需反归一化；若有mean/std，替换下面的值
    mean = torch.tensor([0.5]).view(1,1,1,1).to(x.device)
    std = torch.tensor([0.5]).view(1,1,1,1).to(x.device)
    return x * std + mean

def denormalize(x):
    """
    如果你用了 ImageNet normalize，这一步必须要
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(x.device)
    return x * std + mean


def visualize_frequency_rgb(x, low_ratio=0.4, save_dir="./vis", prefix="rgb"):
    """
    x: [1, 3, H, W]  —— 必须是原始RGB输入！！
    """

    os.makedirs(save_dir, exist_ok=True)

    x = x.detach()

    # ⭐ 如果你数据是normalize过的，打开这一行
    x = denormalize(x)

    x = x.clamp(0, 1)

    x = x.cpu()[0]  # [3,H,W]
    C, H, W = x.shape

    # =========================
    # 原图
    # =========================
    original = x.permute(1,2,0).numpy()

    # =========================
    # DCT
    # =========================
    freq = DCT.dct_2d(x, norm='ortho')

    # =========================
    # ✅ 正确低频mask（左上角）
    # =========================
    h_cut = int(H * low_ratio)
    w_cut = int(W * low_ratio)

    low_mask = torch.zeros(H, W)
    low_mask[:h_cut, :w_cut] = 1.0
    high_mask = 1.0 - low_mask

    low_mask = low_mask.unsqueeze(0)   # [1,H,W]
    high_mask = high_mask.unsqueeze(0)

    # =========================
    # 分离
    # =========================
    freq_low = freq * low_mask
    freq_high = freq * high_mask

    # =========================
    # 重建
    # =========================
    low_spatial = DCT.idct_2d(freq_low, norm='ortho')
    high_spatial = DCT.idct_2d(freq_high, norm='ortho')

    # =========================
    # 增强（关键！！）
    # =========================
    def enhance(x):
        x = x - x.mean()
        x = x / (x.std() + 1e-6)
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        return x

    low_img = enhance(low_spatial)
    high_img = enhance(high_spatial)

    low_img = low_img.permute(1,2,0).numpy()
    high_img = high_img.permute(1,2,0).numpy()

    # =========================
    # 频谱（仅展示一个通道）
    # =========================
    spectrum = torch.log(torch.abs(freq[0]) + 1e-6).numpy()

    # =========================
    # 保存
    # =========================
    plt.imsave(os.path.join(save_dir, f"{prefix}_original.png"), original)
    plt.imsave(os.path.join(save_dir, f"{prefix}_low.png"), low_img)
    plt.imsave(os.path.join(save_dir, f"{prefix}_high.png"), high_img)
    # 在原保存mask的代码后，新增保存high_mask的行
    plt.imsave(os.path.join(save_dir, f"{prefix}_low_mask.png"), low_mask[0].numpy(), cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{prefix}_high_mask.png"), high_mask[0].numpy(), cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{prefix}_spectrum.png"), spectrum, cmap='jet')

    print(f">>> Saved to {save_dir}")

def visualize_frequency_dsm(x, low_ratio=0.4, save_dir="./vis", prefix="dsm"):
    """
    x: [1, 1, H, W]  —— DSM单通道输入
    """
    os.makedirs(save_dir, exist_ok=True)
    x = x.detach()

    # ⭐ DSM反归一化（根据实际情况调整）
    x = denormalize_dsm(x)
    x = x.clamp(0, 1)
    x = x.cpu()[0]  # [1,H,W]
    C, H, W = x.shape

    # =========================
    # 原图（单通道转3通道便于保存）
    # =========================
    original = x.squeeze(0).numpy()  # [H,W]

    # =========================
    # DCT
    # =========================
    freq = DCT.dct_2d(x, norm='ortho')  # [1,H,W]

    # =========================
    # 低频mask（和RGB用相同比例）
    # =========================
    h_cut = int(H * low_ratio)
    w_cut = int(W * low_ratio)

    low_mask = torch.zeros(H, W)
    low_mask[:h_cut, :w_cut] = 1.0
    high_mask = 1.0 - low_mask

    low_mask = low_mask.unsqueeze(0)   # [1,H,W]
    high_mask = high_mask.unsqueeze(0)

    # =========================
    # 分离
    # =========================
    freq_low = freq * low_mask
    freq_high = freq * high_mask

    # =========================
    # 重建
    # =========================
    low_spatial = DCT.idct_2d(freq_low, norm='ortho')
    high_spatial = DCT.idct_2d(freq_high, norm='ortho')

    # =========================
    # 增强（和RGB保持一致逻辑）
    # =========================
    def enhance(x):
        x = x - x.mean()
        x = x / (x.std() + 1e-6)
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        return x

    low_img = enhance(low_spatial).squeeze(0).numpy()  # [H,W]
    high_img = enhance(high_spatial).squeeze(0).numpy()  # [H,W]

    # =========================
    # 频谱
    # =========================
    spectrum = torch.log(torch.abs(freq[0]) + 1e-6).numpy()  # [H,W]

    # =========================
    # 保存（单通道用灰度图）
    # =========================
    plt.imsave(os.path.join(save_dir, f"{prefix}_original.png"), original, cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{prefix}_low.png"), low_img, cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{prefix}_high.png"), high_img, cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{prefix}_low_mask.png"), low_mask[0].numpy(), cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{prefix}_high_mask.png"), high_mask[0].numpy(), cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{prefix}_spectrum.png"), spectrum, cmap='jet')

    print(f">>> DSM可视化保存至 {save_dir}")

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


class Baseline(nn.Module):
    def __init__(self, cfg=None, num_classes=None, norm_layer=nn.BatchNorm2d, in_chans=None):
        super(Baseline, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        if in_chans is not None:
            self.in_chans = in_chans
        else:
            self.in_chans = [3, 1]

        if cfg.backbone == 'mit_b5':
            from .encoder_agg import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b4':
            from .encoder_agg import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b2':
            from .encoder_agg import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b1':
            from .encoder_agg import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b0':
            from .encoder_agg import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
            self.channels = [32, 64, 160, 256]
        else:
            from .encoder_agg import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)

        from .Seg_head import DecoderHead
        self.decode_head = DecoderHead(in_channels=self.channels, num_classes=num_classes, norm_layer=norm_layer,
                                       embed_dim=cfg.decoder_embed_dim)

        self.init_weights(cfg, pretrained=cfg.pretrained_backbone)


    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        ori_size = rgb.shape
        x_semantic, L_cons, low_L_cons = self.backbone(rgb, modal_x)

        out_semantic = self.decode_head.forward(x_semantic)
        out_semantic = F.interpolate(out_semantic, size=ori_size[2:], mode='bilinear', align_corners=False)

        return out_semantic, L_cons, low_L_cons

    def forward(self, rgb, modal_x):
        # ⭐ 这里才是真正的原始输入
        if not hasattr(self, "vis_done"):
            print(">>> VIS INPUT RGB:", rgb.shape)
            print(">>> VIS INPUT DSM:", modal_x.shape)

            rgb_vis = rgb[:, :3, :, :]   # 只取前3个通道
            visualize_frequency_rgb(rgb_vis, save_dir="./vis", prefix="rgb")

            # DSM可视化（先确保维度正确）
            if modal_x.ndim == 3:
                modal_x = torch.unsqueeze(modal_x, dim=1)  # [B,1,H,W]
            dsm_vis = modal_x[:, :1, :, :]  # 只取第一个通道（DSM单通道）
            visualize_frequency_dsm(dsm_vis, save_dir="./vis", prefix="dsm")

            self.vis_done = True
        if modal_x.ndim == 3:
            modal_x = torch.unsqueeze(modal_x, dim=1)
        outputs, L_cons, low_L_cons = self.encode_decode(rgb, modal_x)

        return outputs, L_cons, low_L_cons
