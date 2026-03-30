import math
import time
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from .acfm import AdaptiveCrossFrequencyModule
from .cmsg import CrossModalStructureGuidance
from .uaf import UncertaintyAwareFusion

# === 修复后的可视化函数 ===
import matplotlib.pyplot as plt
import torch_dct as DCT
import os

def visualize_acfm_effect(rgb, dsm, acfm_module, save_dir="./vis_results"):
    os.makedirs(save_dir, exist_ok=True)
    # 1. 原始特征的频域振幅差（ACFM前）
    freq_rgb_ori = DCT.dct_2d(rgb, norm='ortho')
    freq_dsm_ori = DCT.dct_2d(dsm, norm='ortho')
    amp_diff_ori = torch.abs(freq_rgb_ori - freq_dsm_ori).mean(dim=1, keepdim=True)
    
    # 2. ACFM处理后的振幅差
    rgb_freq = acfm_module.freq_transform(rgb)
    dsm_freq = acfm_module.freq_transform(dsm)
    freq_rgb_new = DCT.dct_2d(rgb_freq, norm='ortho')
    freq_dsm_new = DCT.dct_2d(dsm_freq, norm='ortho')
    amp_diff_new = torch.abs(freq_rgb_new - freq_dsm_new).mean(dim=1, keepdim=True)
    
    # 3. 门控权重可视化（g_low/g_high）
    g = torch.sigmoid(acfm_module.gate(rgb))  # [B,2,1,1]
    g_low = g[:,0].mean().item()
    g_high = g[:,1].mean().item()
    
    # 4. 画图对比（关键修复：detach() 脱离计算图）
    plt.figure(figsize=(10,4))
    # 原始振幅差
    plt.subplot(1,2,1)
    plt.imshow(amp_diff_ori[0,0].detach().cpu().numpy(), cmap='viridis')  # 修复处
    plt.title(f"Before ACFM (g_low={g_low:.2f}, g_high={g_high:.2f})")
    plt.axis('off')
    # ACFM后振幅差
    plt.subplot(1,2,2)
    plt.imshow(amp_diff_new[0,0].detach().cpu().numpy(), cmap='viridis')  # 修复处
    plt.title("After ACFM (Reduced Inter-modal Difference)")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "acfm_amp_diff.png"), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_cmsg_effect(rgb, dsm, cmsg_module, save_dir="./vis_results"):
    os.makedirs(save_dir, exist_ok=True)
    # 1. 提取DSM高频结构
    blur = cmsg_module.dsm_blur(dsm)
    structure = dsm - blur
    structure_refine = cmsg_module.structure_refine(structure)
    
    # 2. 注意力权重
    a = cmsg_module.attn(torch.cat([rgb, structure_refine], dim=1))
    
    # 3. 可视化（关键修复：detach()）
    plt.figure(figsize=(12,3))
    # DSM原图
    plt.subplot(1,4,1)
    plt.imshow(dsm[0,0].detach().cpu().numpy(), cmap='gray')  # 修复处
    plt.title("DSM Original")
    plt.axis('off')
    # DSM高频结构
    plt.subplot(1,4,2)
    plt.imshow(structure_refine[0,0].detach().cpu().numpy(), cmap='gray')  # 修复处
    plt.title("DSM High-pass Structure")
    plt.axis('off')
    # 注意力权重
    plt.subplot(1,4,3)
    plt.imshow(a[0,0].detach().cpu().numpy(), cmap='jet')  # 修复处
    plt.title("CMSG Attention Weight")
    plt.axis('off')
    # 引导后的RGB
    guided_rgb = rgb + rgb * a
    plt.subplot(1,4,4)
    plt.imshow(guided_rgb[0,0].detach().cpu().numpy(), cmap='gray')  # 修复处
    plt.title("Guided RGB")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "cmsg_structure.png"), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_uaf_effect(rgb, dsm, uaf_module, save_dir="./vis_results"):
    os.makedirs(save_dir, exist_ok=True)
    # 1. 置信度分数和权重
    rgb_score = uaf_module.rgb_conf(rgb)
    dsm_score = uaf_module.dsm_conf(dsm)
    scores = torch.cat([rgb_score, dsm_score], dim=1)
    weights = F.softmax(scores / uaf_module.temperature, dim=1)
    w_rgb = weights[:,0:1]
    w_dsm = weights[:,1:2]
    
    # 2. 融合特征
    fusion = w_rgb * rgb + w_dsm * dsm
    
    # 3. 可视化（关键修复：detach()）
    plt.figure(figsize=(12,3))
    # RGB置信度
    plt.subplot(1,4,1)
    plt.imshow(w_rgb[0,0].detach().cpu().numpy(), cmap='jet')  # 修复处
    plt.title("RGB Confidence Weight")
    plt.axis('off')
    # DSM置信度
    plt.subplot(1,4,2)
    plt.imshow(w_dsm[0,0].detach().cpu().numpy(), cmap='jet')  # 修复处
    plt.title("DSM Confidence Weight")
    plt.axis('off')
    # 原始RGB（模糊区）
    plt.subplot(1,4,3)
    plt.imshow(rgb[0,0].detach().cpu().numpy(), cmap='gray')  # 修复处
    plt.title("Original RGB (Blurred)")
    plt.axis('off')
    # 融合后特征
    plt.subplot(1,4,4)
    plt.imshow(fusion[0,0].detach().cpu().numpy(), cmap='gray')  # 修复处
    plt.title("UAF Fusion (Sharpened)")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "uaf_confidence.png"), dpi=300, bbox_inches='tight')
    plt.close()

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SimpleFusion(nn.Module):
    """
    Baseline fusion (no ASD / no LFGF / no CFI):
    concat(rgb_feat, extra_feat) -> 1x1 conv -> fused_feat
    """
    def __init__(self, c: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_rgb, x_e], dim=1)
        return self.proj(x)


class RGBXTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=None, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        if in_chans is not None:
            self.in_chans = in_chans
        else:
            raise ValueError('in_chans should not be None')

        # patch_embed (rgb)
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=self.in_chans[0],
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # patch_embed (extra)
        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4,
                                                    in_chans=self.in_chans[1],
                                                    embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                    in_chans=embed_dims[0],
                                                    embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                    in_chans=embed_dims[1],
                                                    embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])
        self.acfm4 = AdaptiveCrossFrequencyModule(channels=embed_dims[3], low_radius=0.35)
        
        self.cmsg1 = CrossModalStructureGuidance(embed_dims[0])
        self.cmsg2 = CrossModalStructureGuidance(embed_dims[1])
        self.cmsg3 = CrossModalStructureGuidance(embed_dims[2])
        self.cmsg4 = CrossModalStructureGuidance(embed_dims[3])
        
        self.uaf1 = UncertaintyAwareFusion(embed_dims[0])
        self.uaf2 = UncertaintyAwareFusion(embed_dims[1])
        self.uaf3 = UncertaintyAwareFusion(embed_dims[2])
        self.uaf4 = UncertaintyAwareFusion(embed_dims[3])

        self.vis_done = False

        # transformer encoder blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])

        self.fuse1 = SimpleFusion(c=embed_dims[0])

        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])

        self.fuse2 = SimpleFusion(c=embed_dims[1])

        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])

        self.fuse3 = SimpleFusion(c=embed_dims[2])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])

        self.fuse4 = SimpleFusion(c=embed_dims[3])

        self.apply(self._init_weights)

    def fusion_loss(self, rgb, dsm):
        """
        计算结合ACFM, CMSG, UAF的损失。
        输出两个标量：L_cons, low_L_cons
        """
        # 1. 获取ACFM增强特征
        acfm_feat = self.acfm4(rgb, dsm)

        # 2. 获取CMSG增强特征
        cmsg_feat = self.cmsg4(rgb, dsm)

        # 3. 获取UAF融合特征
        uaf_feat = self.uaf4(rgb, dsm)

        # 简单设计：使用L1损失衡量各特征之间的一致性
        loss_acfm_uaf = F.l1_loss(acfm_feat, uaf_feat)
        loss_cmsg_uaf = F.l1_loss(cmsg_feat, uaf_feat)

        # 合并作为 L_cons
        L_cons = loss_acfm_uaf + loss_cmsg_uaf

        # low_L_cons 可以用低频信息损失（ACFM低频与高频分离）
        # 取ACFM的低频 mask
        B, C, H, W = rgb.shape
        low_mask = self.acfm4._build_low_mask(H, W, rgb.device, rgb.dtype)
        low_freq_rgb = rgb * low_mask
        low_freq_dsm = dsm * low_mask
        low_L_cons = F.mse_loss(low_freq_rgb, low_freq_dsm)

        return L_cons, low_L_cons

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained, self.in_chans)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x_rgb, x_e):
        B = x_rgb.shape[0]
        outs_semantic = []

        # Stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        x_e, _, _ = self.extra_patch_embed1(x_e)
        for blk in self.block1:
            x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block1:
            x_e = blk(x_e, H, W)
        x_rgb = self.norm1(x_rgb)
        x_e = self.extra_norm1(x_e)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb = self.cmsg1(x_rgb, x_e)

        outs_semantic.append(self.uaf1(x_rgb, x_e))

        # fused = self.uaf1(x_rgb, x_e)
        # if isinstance(fused, (tuple, list)):
        #     fused = fused[0]
        # outs_semantic.append(fused)

        # outs_semantic.append(self.fuse1(x_rgb, x_e))

        # Stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_e, _, _ = self.extra_patch_embed2(x_e)
        for blk in self.block2:
            x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block2:
            x_e = blk(x_e, H, W)
        x_rgb = self.norm2(x_rgb)
        x_e = self.extra_norm2(x_e)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb = self.cmsg2(x_rgb, x_e)
        outs_semantic.append(self.uaf2(x_rgb, x_e))

        # fused = self.uaf2(x_rgb, x_e)
        # if isinstance(fused, (tuple, list)):
        #     fused = fused[0]
        # outs_semantic.append(fused)

        # outs_semantic.append(self.fuse2(x_rgb, x_e))

        # Stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_e, _, _ = self.extra_patch_embed3(x_e)
        for blk in self.block3:
            x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block3:
            x_e = blk(x_e, H, W)
        x_rgb = self.norm3(x_rgb)
        x_e = self.extra_norm3(x_e)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb = self.cmsg3(x_rgb, x_e)
        outs_semantic.append(self.uaf3(x_rgb, x_e))

        # fused = self.uaf3(x_rgb, x_e)
        # if isinstance(fused, (tuple, list)):
        #     fused = fused[0]
        # outs_semantic.append(fused)

        # outs_semantic.append(self.fuse3(x_rgb, x_e))

        # Stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_e, _, _ = self.extra_patch_embed4(x_e)
        for blk in self.block4:
            x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block4:
            x_e = blk(x_e, H, W)
        x_rgb = self.norm4(x_rgb)
        x_e = self.extra_norm4(x_e)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb = self.cmsg4(x_rgb, x_e)
        visualize_acfm_effect(x_rgb, x_e, self.acfm4)

        # === 新增：运行可视化（只运行一次） ===
        # if not self.vis_done:
        #     # ACFM可视化（Stage4特征）
        #     visualize_acfm_effect(x_rgb, x_e, self.acfm4)
        #     # CMSG可视化（Stage4特征）
        #     visualize_cmsg_effect(x_rgb, x_e, self.cmsg4)
        #     # UAF可视化（Stage4特征）
        #     visualize_uaf_effect(x_rgb, x_e, self.uaf4)
        #     self.vis_done = True  # 标记完成，避免重复运行

        x_rgb = self.acfm4(x_rgb, x_e)  # inject DSM freq info into RGB (stage4 only)
        
        outs_semantic.append(self.uaf4(x_rgb, x_e))
        # fused = self.uaf4(x_rgb, x_e)
        # if isinstance(fused, (tuple, list)):
        #     fused = fused[0]
        # outs_semantic.append(fused)

        # outs_semantic.append(self.fuse4(x_rgb, x_e))


        # keep interface: return (outs, L_cons, low_L_cons)
        # device = outs_semantic[-1].device
        # L_cons = torch.zeros([], device=device)
        # low_L_cons = torch.zeros([], device=device)
        # return outs_semantic, L_cons, low_L_cons
        last = outs_semantic[-1]
        if isinstance(last, (tuple, list)):
            last = last[0]
        device = last.device

        # 建议用 shape [1]，避免 DataParallel gather scalar 警告
        L_cons = last.new_zeros(1)
        low_L_cons = last.new_zeros(1)
        return outs_semantic, L_cons, low_L_cons

    def forward(self, x_rgb, x_e):
        out_semantic, L_cons, low_L_cons = self.forward_features(x_rgb, x_e)
        # return out_semantic, L_cons, low_L_cons
        last = out_semantic[-1]
        if isinstance(last, (tuple, list)):
            last = last[0]

        L_cons, low_L_cons = self.fusion_loss(last, last)  # 这里可以传当前RGB和DSM
        return out_semantic, L_cons, low_L_cons


def load_dualpath_model(model, model_file, in_chans):
    t0 = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            if k.find('patch_embed1.proj.weight') >= 0:
                v = _adapt_first_conv(v, in_chans[0])
                extra_v = _adapt_first_conv(v, in_chans[1])
                state_dict[k] = v
                state_dict[k.replace('patch_embed1', 'extra_patch_embed1')] = extra_v
            else:
                state_dict[k] = v
                state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
            state_dict[k.replace('block', 'extra_block')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'extra_norm')] = v

    t_io = time.time()
    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict

    t_end = time.time()
    miss, unexp = len(msg.missing_keys), len(msg.unexpected_keys)
    print(f"[load_dualpath_model] IO {t_io - t0:.2f}s | load {t_end - t_io:.2f}s")
    print(f"  missing={miss}  unexpected={unexp}")
    if miss:
        print("  first 10 missing:", msg.missing_keys[:10])
    if unexp:
        print("  first 10 unexpected:", msg.unexpected_keys[:10])


def _adapt_first_conv(weight, in_chans: int):
    if weight.shape[1] == in_chans:
        return weight

    if in_chans < 3:
        new_weight = weight.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
    else:
        repeat = math.ceil(in_chans / 3)
        new_weight = weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :].clone()
    return new_weight


class mit_b0(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=None)


class mit_b1(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=None)


class mit_b2(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=None)


class mit_b3(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=None)


class mit_b4(RGBXTransformer):
    def __init__(self, in_chans, fuse_cfg=None, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=in_chans)


class mit_b5(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=None)