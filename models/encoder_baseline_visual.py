import math
import time
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from datetime import datetime
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 核心创新模块导入
from .acfm import AdaptiveCrossFrequencyModule
from .cmsg import CrossModalStructureGuidance
from .uaf import UncertaintyAwareFusion

# 尝试导入 DCT 库
try:
    import torch_dct as DCT
except ImportError:
    print("[Warning] torch_dct not found. ACFM visualization might fail.")

# ==========================================
# 1. 顶刊级可视化辅助工具 (Visual Pro Helpers)
# ==========================================

def _get_unique_filename(prefix, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(save_dir, f"{prefix}_{timestamp}.png")

def _tensor_to_np_img(t, is_rgb=True):
    """将 Tensor [B, C, H, W] 转为 HWC 的 numpy 数组 [0, 255]"""
    img = t.detach().cpu()[0]
    if is_rgb:
        img = img[:3].permute(1, 2, 0).numpy()
    else:
        img = img[0].numpy() if img.dim() == 3 else img.numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return (img * 255).astype(np.uint8)

def _apply_heatmap_overlay(weight_map, base_img_rgb):
    """将权重图平滑上采样并叠加在原图上"""
    h, w = base_img_rgb.shape[:2]
    # 核心：使用双线性插值平滑上采样
    heatmap_np = F.interpolate(weight_map, size=(h, w), mode='bilinear', align_corners=False).detach().cpu()[0, 0].numpy()
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-6)
    heatmap_color = cv2.applyColorMap((heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_img_rgb, 0.5, heatmap_rgb, 0.5, 0)
    return heatmap_rgb, overlay

# ==========================================
# 2. 核心可视化逻辑 (CMSG, UAF, ACFM)
# ==========================================

def visualize_cmsg_pro(raw_rgb, raw_dsm, cmsg_module, f_r, f_d, save_dir):
    save_path = os.path.join(save_dir, "cmsg")
    try:
        with torch.no_grad():
            bg_rgb = _tensor_to_np_img(raw_rgb, is_rgb=True)
            bg_dsm = _tensor_to_np_img(raw_dsm, is_rgb=False)
            blur = cmsg_module.dsm_blur(f_d)
            struct = f_d - blur
            if hasattr(cmsg_module, 'structure_refine'):
                struct = cmsg_module.structure_refine(struct)
            attn = cmsg_module.attn(torch.cat([f_r, struct], dim=1))
            h_map, overlay = _apply_heatmap_overlay(attn, bg_rgb)
            
            plt.figure(figsize=(20, 4))
            plt.subplot(1, 5, 1); plt.imshow(bg_rgb); plt.title("Input RGB"); plt.axis('off')
            plt.subplot(1, 5, 2); plt.imshow(bg_dsm, cmap='gray'); plt.title("Input DSM"); plt.axis('off')
            plt.subplot(1, 5, 3); plt.imshow(_tensor_to_np_img(struct, False), cmap='gray'); plt.title("Edges"); plt.axis('off')
            plt.subplot(1, 5, 4); plt.imshow(h_map); plt.title("Attention Map"); plt.axis('off')
            plt.subplot(1, 5, 5); plt.imshow(overlay); plt.title("Structural Guidance"); plt.axis('off')
            plt.savefig(_get_unique_filename("cmsg_pro", save_path), dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e: print(f"[Vis Error] CMSG: {e}")

def visualize_uaf_pro(raw_rgb, raw_dsm, uaf_module, f_r, f_d, save_dir):
    save_path = os.path.join(save_dir, "uaf")
    try:
        with torch.no_grad():
            bg_rgb = _tensor_to_np_img(raw_rgb, is_rgb=True)
            bg_dsm = _tensor_to_np_img(raw_dsm, is_rgb=False)
            r_s, d_s = uaf_module.rgb_conf(f_r), uaf_module.dsm_conf(f_d)
            temp = getattr(uaf_module, 'temperature', 1.0)
            weights = F.softmax(torch.cat([r_s, d_s], dim=1) / temp, dim=1)
            h_r, _ = _apply_heatmap_overlay(weights[:, 0:1], bg_rgb)
            h_d, over_d = _apply_heatmap_overlay(weights[:, 1:2], bg_rgb)
            
            plt.figure(figsize=(20, 4))
            plt.subplot(1, 5, 1); plt.imshow(bg_rgb); plt.title("Input RGB"); plt.axis('off')
            plt.subplot(1, 5, 2); plt.imshow(bg_dsm, cmap='gray'); plt.title("Input DSM"); plt.axis('off')
            plt.subplot(1, 5, 3); plt.imshow(h_r); plt.title("RGB Confidence"); plt.axis('off')
            plt.subplot(1, 5, 4); plt.imshow(h_d); plt.title("DSM Confidence"); plt.axis('off')
            plt.subplot(1, 5, 5); plt.imshow(over_d); plt.title("Final Arbitration"); plt.axis('off')
            plt.savefig(_get_unique_filename("uaf_pro", save_path), dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e: print(f"[Vis Error] UAF: {e}")

def visualize_acfm_pro(raw_rgb, raw_dsm, acfm_module, f_r, f_d, save_dir):
    save_path = os.path.join(save_dir, "acfm")
    try:
        with torch.no_grad():
            f_r_o, f_d_o = DCT.dct_2d(f_r, norm='ortho'), DCT.dct_2d(f_d, norm='ortho')
            diff_o = torch.abs(f_r_o - f_d_o).mean(dim=1, keepdim=True)
            r_aligned = acfm_module.freq_transform(f_r)
            d_aligned = acfm_module.freq_transform(f_d)
            diff_new = torch.abs(DCT.dct_2d(r_aligned) - DCT.dct_2d(d_aligned)).mean(dim=1, keepdim=True)
            g = torch.sigmoid(acfm_module.gate(f_r))
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(_tensor_to_np_img(diff_o, False), cmap='magma')
            plt.title(f"Before Alignment\nG_low:{g[0,0].item():.2f} G_high:{g[0,1].item():.2f}"); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(_tensor_to_np_img(diff_new, False), cmap='magma')
            plt.title("After Alignment"); plt.axis('off')
            plt.savefig(_get_unique_filename("acfm_pro", save_path), dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e: print(f"[Vis Error] ACFM: {e}")

# ==========================================
# 3. Transformer 基础组件
# ==========================================

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return self.dwconv(x).flatten(2).transpose(1, 2)

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
    def forward(self, x, H, W):
        x = self.fc1(x); x = self.dwconv(x, H, W); x = self.act(x)
        return self.drop(self.fc2(self.drop(x)))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop, self.proj = nn.Dropout(attn_drop), nn.Linear(dim, dim)
        self.proj_drop, self.sr_ratio = nn.Dropout(proj_drop), sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.norm(self.sr(x_).reshape(B, C, -1).permute(0, 2, 1))
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        return self.proj_drop(self.proj((attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride, patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x); _, _, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W

# ==========================================
# 4. RGBXTransformer 主类 (修正返回值逻辑)
# ==========================================

class RGBXTransformer(nn.Module):
    def __init__(self, img_size=256, in_chans=[3, 1], embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.in_chans = in_chans
        self.vis_done = False
        self.save_dir = "./vis_pro" 

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            ic, ec = (in_chans[0] if i==0 else embed_dims[i-1]), (in_chans[1] if i==0 else embed_dims[i-1])
            ps, st = (7, 4) if i==0 else (3, 2)
            setattr(self, f"patch_embed{i+1}", OverlapPatchEmbed(img_size//(2**i), ps, st, ic, embed_dims[i]))
            setattr(self, f"extra_patch_embed{i+1}", OverlapPatchEmbed(img_size//(2**i), ps, st, ec, embed_dims[i]))
            
            setattr(self, f"block{i+1}", nn.ModuleList([Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+j], norm_layer=nn.LayerNorm, sr_ratio=sr_ratios[i]) for j in range(depths[i])]))
            setattr(self, f"norm{i+1}", nn.LayerNorm(embed_dims[i]))
            setattr(self, f"extra_block{i+1}", nn.ModuleList([Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+j], norm_layer=nn.LayerNorm, sr_ratio=sr_ratios[i]) for j in range(depths[i])]))
            setattr(self, f"extra_norm{i+1}", nn.LayerNorm(embed_dims[i]))
            
            setattr(self, f"cmsg{i+1}", CrossModalStructureGuidance(embed_dims[i]))
            setattr(self, f"uaf{i+1}", UncertaintyAwareFusion(embed_dims[i]))
            cur += depths[i]
        
        self.acfm4 = AdaptiveCrossFrequencyModule(embed_dims[3])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained, self.in_chans)

    def forward_features(self, x_rgb, x_dsm):
        raw_r, raw_d = x_rgb.clone(), x_dsm.clone()
        B = x_rgb.shape[0]; outs = []
        for i in range(4):
            x_rgb, H, W = getattr(self, f"patch_embed{i+1}")(x_rgb)
            x_dsm, _, _ = getattr(self, f"extra_patch_embed{i+1}")(x_dsm)
            for blk in getattr(self, f"block{i+1}"): x_rgb = blk(x_rgb, H, W)
            for blk in getattr(self, f"extra_block{i+1}"): x_dsm = blk(x_dsm, H, W)
            x_rgb = getattr(self, f"norm{i+1}")(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_dsm = getattr(self, f"extra_norm{i+1}")(x_dsm).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            
            if not self.vis_done:
                if i == 0:
                    visualize_cmsg_pro(raw_r, raw_d, getattr(self, f"cmsg{i+1}"), x_rgb, x_dsm, self.save_dir)
                    visualize_uaf_pro(raw_r, raw_d, getattr(self, f"uaf{i+1}"), x_rgb, x_dsm, self.save_dir)
                if i == 3:
                    visualize_acfm_pro(raw_r, raw_d, self.acfm4, x_rgb, x_dsm, self.save_dir); 
            
            x_rgb = getattr(self, f"cmsg{i+1}")(x_rgb, x_dsm)
            if i == 3: x_rgb = self.acfm4(x_rgb, x_dsm)
            outs.append(getattr(self, f"uaf{i+1}")(x_rgb, x_dsm))
        
        # 返回列表形式的 outs，供解码器头解包
        return outs, torch.zeros(1, device=x_rgb.device), torch.zeros(1, device=x_rgb.device)

    def forward(self, x_rgb, x_dsm):
        # 【核心修正】：这里必须返回完整的列表 outs (包含 4 个层级的特征)
        outs, L, low = self.forward_features(x_rgb, x_dsm)
        return outs, L, low

# ==========================================
# 5. 其他配套子类与工具
# ==========================================

def load_dualpath_model(model, model_file, in_chans):
    print(f"[Loading] Backbone pretraining from {model_file}")
    raw_dict = torch.load(model_file, map_location='cpu')
    if 'model' in raw_dict: raw_dict = raw_dict['model']
    state_dict = {}
    for k, v in raw_dict.items():
        if 'patch_embed' in k:
            if 'patch_embed1.proj.weight' in k:
                state_dict[k] = _adapt_conv(v, in_chans[0])
                state_dict[k.replace('patch_embed', 'extra_patch_embed')] = _adapt_conv(v, in_chans[1])
            else:
                state_dict[k] = v; state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
        elif 'block' in k:
            state_dict[k] = v; state_dict[k.replace('block', 'extra_block')] = v
        elif 'norm' in k:
            state_dict[k] = v; state_dict[k.replace('norm', 'extra_norm')] = v
    model.load_state_dict(state_dict, strict=False)

def _adapt_conv(weight, ichan):
    if weight.shape[1] == ichan: return weight
    if ichan < 3: return weight.mean(dim=1, keepdim=True).repeat(1, ichan, 1, 1)
    return weight.repeat(1, math.ceil(ichan/3), 1, 1)[:, :ichan, :, :].clone()

class mit_b4(RGBXTransformer):
    def __init__(self, in_chans=[3, 1], **kwargs):
        super().__init__(img_size=256, in_chans=in_chans, depths=[3, 8, 27, 3], embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8])