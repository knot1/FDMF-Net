import os
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from skimage import io
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 请确保你的模型组件路径正确
from models.acfm import AdaptiveCrossFrequencyModule
from models.cmsg import CrossModalStructureGuidance
from models.uaf import UncertaintyAwareFusion

try:
    import torch_dct as DCT
except ImportError:
    print("[Warning] torch_dct not found.")

# ==========================================
# 1. 顶刊级可视化辅助工具 (Visual Pro Helpers)
# ==========================================

def _tensor_to_np_img(t, is_rgb=True):
    img = t.detach().cpu()[0]
    if is_rgb:
        img = img[:3].permute(1, 2, 0).numpy()
    else:
        img = img[0].numpy() if img.dim() == 3 else img.numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return (img * 255).astype(np.uint8)

def _apply_heatmap_overlay(weight_map, base_img_rgb):
    h, w = base_img_rgb.shape[:2]
    heatmap_np = F.interpolate(weight_map, size=(h, w), mode='bilinear', align_corners=False).detach().cpu()[0, 0].numpy()
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-6)
    heatmap_color = cv2.applyColorMap((heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_img_rgb, 0.5, heatmap_rgb, 0.5, 0)
    return heatmap_rgb, overlay

def _get_unique_fn(prefix, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(save_dir, f"{prefix}_{ts}.png")

# ==========================================
# 2. 可视化机制函数 (CMSG, UAF, ACFM)
# ==========================================

def visualize_cmsg_pro(raw_rgb, raw_dsm, cmsg_module, f_r, f_d, save_dir):
    try:
        with torch.no_grad():
            bg_rgb = _tensor_to_np_img(raw_rgb, True)
            bg_dsm = _tensor_to_np_img(raw_dsm, False)
            blur = cmsg_module.dsm_blur(f_d)
            struct = f_d - blur
            if hasattr(cmsg_module, 'structure_refine'):
                struct = cmsg_module.structure_refine(struct)
            attn = cmsg_module.attn(torch.cat([f_r, struct], dim=1))
            h_map, overlay = _apply_heatmap_overlay(attn, bg_rgb)
            plt.figure(figsize=(20, 4))
            plt.subplot(1,5,1); plt.imshow(bg_rgb); plt.title("Input RGB"); plt.axis('off')
            plt.subplot(1,5,2); plt.imshow(bg_dsm, cmap='gray'); plt.title("Input DSM"); plt.axis('off')
            plt.subplot(1,5,3); plt.imshow(_tensor_to_np_img(struct, False), cmap='gray'); plt.title("Edges"); plt.axis('off')
            plt.subplot(1,5,4); plt.imshow(h_map); plt.title("Attention Map"); plt.axis('off')
            plt.subplot(1,5,5); plt.imshow(overlay); plt.title("Structural Guidance"); plt.axis('off')
            plt.savefig(_get_unique_fn("cmsg_pro", os.path.join(save_dir, "cmsg")), dpi=300); plt.close()
    except Exception as e: print(f"CMSG Vis Error: {e}")

def visualize_uaf_pro(raw_rgb, uaf_module, f_r, f_d, save_dir):
    try:
        with torch.no_grad():
            bg_rgb = _tensor_to_np_img(raw_rgb, True)
            r_s, d_s = uaf_module.rgb_conf(f_r), uaf_module.dsm_conf(f_d)
            temp = getattr(uaf_module, 'temperature', 1.0)
            weights = F.softmax(torch.cat([r_s, d_s], 1) / temp, 1)
            w_r, w_d = weights[:, 0:1], weights[:, 1:2]
            h_r, _ = _apply_heatmap_overlay(w_r, bg_rgb)
            h_d, over_d = _apply_heatmap_overlay(w_d, bg_rgb)
            plt.figure(figsize=(16, 4))
            plt.subplot(1,4,1); plt.imshow(bg_rgb); plt.title("Input RGB"); plt.axis('off')
            plt.subplot(1,4,2); plt.imshow(h_r); plt.title("RGB Confidence"); plt.axis('off')
            plt.subplot(1,4,3); plt.imshow(h_d); plt.title("DSM Confidence"); plt.axis('off')
            plt.subplot(1,4,4); plt.imshow(over_d); plt.title("Final Arbitration"); plt.axis('off')
            plt.savefig(_get_unique_fn("uaf_pro", os.path.join(save_dir, "uaf")), dpi=300); plt.close()
    except Exception as e: print(f"UAF Vis Error: {e}")

def visualize_acfm_pro(acfm_module, f_r, f_d, save_dir):
    try:
        with torch.no_grad():
            f_r_o, f_d_o = DCT.dct_2d(f_r, norm='ortho'), DCT.dct_2d(f_d, norm='ortho')
            diff_o = torch.abs(f_r_o - f_d_o).mean(1, True)
            r_a, d_a = acfm_module.freq_transform(f_r), acfm_module.freq_transform(f_d)
            diff_a = torch.abs(DCT.dct_2d(r_a) - DCT.dct_2d(d_a)).mean(1, True)
            g = torch.sigmoid(acfm_module.gate(f_r))
            plt.figure(figsize=(10, 5))
            plt.subplot(1,2,1); plt.imshow(_tensor_to_np_img(diff_o, False), cmap='magma')
            plt.title(f"Before Alignment\nG_high:{g[0,1].item():.2f}"); plt.axis('off')
            plt.subplot(1,2,2); plt.imshow(_tensor_to_np_img(diff_a, False), cmap='magma')
            plt.title("After Alignment"); plt.axis('off')
            plt.savefig(_get_unique_fn("acfm_pro", os.path.join(save_dir, "acfm")), dpi=300); plt.close()
    except Exception as e: print(f"ACFM Vis Error: {e}")

# ==========================================
# 3. 核心模型组件
# ==========================================

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
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
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

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
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride, patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x); _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W

# ==========================================
# 4. RGBXTransformer 主类
# ==========================================

class RGBXTransformer(nn.Module):
    def __init__(self, img_size=256, in_chans=[3, 1], embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.vis_done = False
        self.save_dir = "./vis_pro_results"

        # 使用关键字参数调用 Block，防止参数位置出错
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            # PatchEmbed
            ichan = in_chans[0] if i==0 else embed_dims[i-1]
            eichan = in_chans[1] if i==0 else embed_dims[i-1]
            ps, st = (7, 4) if i==0 else (3, 2)
            setattr(self, f"patch_embed{i+1}", OverlapPatchEmbed(img_size//(2**i), ps, st, ichan, embed_dims[i]))
            setattr(self, f"extra_patch_embed{i+1}", OverlapPatchEmbed(img_size//(2**i), ps, st, eichan, embed_dims[i]))
            
            # Blocks
            setattr(self, f"block{i+1}", nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, 
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+j], 
                norm_layer=nn.LayerNorm, sr_ratio=sr_ratios[i]) for j in range(depths[i])]))
            setattr(self, f"norm{i+1}", nn.LayerNorm(embed_dims[i]))
            setattr(self, f"extra_block{i+1}", nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, 
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+j], 
                norm_layer=nn.LayerNorm, sr_ratio=sr_ratios[i]) for j in range(depths[i])]))
            setattr(self, f"extra_norm{i+1}", nn.LayerNorm(embed_dims[i]))
            
            # Innovation
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

    def forward_features(self, x_rgb, x_dsm):
        raw_r, raw_d = x_rgb.clone(), x_dsm.clone()
        B = x_rgb.shape[0]
        outs = []

        for i in range(4):
            x_rgb, H, W = getattr(self, f"patch_embed{i+1}")(x_rgb)
            x_dsm, _, _ = getattr(self, f"extra_patch_embed{i+1}")(x_dsm)
            for blk in getattr(self, f"block{i+1}"): x_rgb = blk(x_rgb, H, W)
            for blk in getattr(self, f"extra_block{i+1}"): x_dsm = blk(x_dsm, H, W)
            x_rgb = getattr(self, f"norm{i+1}")(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_dsm = getattr(self, f"extra_norm{i+1}")(x_dsm).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            if i == 0 and not self.vis_done:
                visualize_cmsg_pro(raw_r, raw_d, self.cmsg1, x_rgb, x_dsm, self.save_dir)
                visualize_uaf_pro(raw_r, self.uaf1, x_rgb, x_dsm, self.save_dir)
            if i == 3 and not self.vis_done:
                visualize_acfm_pro(self.acfm4, x_rgb, x_dsm, self.save_dir); self.vis_done = True

            x_rgb = getattr(self, f"cmsg{i+1}")(x_rgb, x_dsm)
            if i == 3: x_rgb = self.acfm4(x_rgb, x_dsm)
            outs.append(getattr(self, f"uaf{i+1}")(x_rgb, x_dsm))
        return outs, torch.zeros(1, device=x_rgb.device), torch.zeros(1, device=x_rgb.device)

    def forward(self, x_rgb, x_dsm):
        outs, _, _ = self.forward_features(x_rgb, x_dsm)
        return outs[-1], torch.zeros(1), torch.zeros(1)

def mit_b4(in_chans=[3, 1]):
    return RGBXTransformer(img_size=256, in_chans=in_chans, depths=[3, 8, 27, 3], embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8])

# ==========================================
# 5. 执行脚本
# ==========================================

def run_visualization_task():
    best_model_path = '/home/wsj/FDMF-Net/Baseline_Vaihingen_42-1/2026-03-16_21-06-21_baseline_vaihingen_innovation123/results_Baseline_vaihingen/best_model_vaihingen'
    data_root = '/data3/wsjdataset/Vaihingen_unzip/ISPRS_semantic_labeling_Vaihingen.zip'
    
    # 修正路径：Vaihingen 常见的 ID 是 1, 3, 5 等整数
    rgb_fmt = data_root + '/top/top_mosaic_09cm_area{}.tif'
    dsm_fmt = data_root + '/dsm/dsm_09cm_matching_area{}.tif'
    test_ids = ['5', '15', '21', '30'] 

    model = mit_b4(in_chans=[3, 1])
    print(f"Loading weights from {best_model_path}")
    state_dict = torch.load(best_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()

    with torch.no_grad():
        for tid in test_ids:
            print(f"Processing Area: {tid}")
            try:
                rgb_img = io.imread(rgb_fmt.format(tid))
                dsm_img = io.imread(dsm_fmt.format(tid))
                
                # 预处理：缩放到 256x256 (或者你的训练尺寸) 
                # 注意：如果直接读全图，内存会爆。这里建议取中间的一个 patch。
                h, w = rgb_img.shape[:2]
                ch, cw = h//2, w//2
                rgb_patch = rgb_img[ch-128:ch+128, cw-128:cw+128]
                dsm_patch = dsm_img[ch-128:ch+128, cw-128:cw+128]

                r_tensor = torch.from_numpy(rgb_patch.astype('float32')/255.).permute(2,0,1).unsqueeze(0).cuda()
                dmin, dmax = dsm_patch.min(), dsm_patch.max()
                d_tensor = torch.from_numpy((dsm_patch.astype('float32')-dmin)/(dmax-dmin+1e-6)).unsqueeze(0).unsqueeze(0).cuda()

                model.vis_done = False
                _ = model(r_tensor, d_tensor)
            except Exception as e:
                print(f"Error processing area{tid}: {e}")
    
    print("Success! Results are in ./vis_pro_results")

if __name__ == "__main__":
    run_visualization_task()