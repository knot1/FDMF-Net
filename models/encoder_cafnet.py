import math
import time
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .acfm import AdaptiveCrossFrequencyModule
from .cmsg import CrossModalStructureGuidance
from .uaf import UncertaintyAwareFusion


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
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
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"

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
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
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
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
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

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
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


class RGBXTransformer(nn.Module):
    def __init__(
    self,
    img_size=256,
    patch_size=16,
    in_chans=None,
    num_classes=1000,
    embed_dims=[64, 128, 256, 512],
    num_heads=[1, 2, 4, 8],
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=False,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=nn.LayerNorm,
    norm_fuse=nn.BatchNorm2d,
    depths=[3, 4, 6, 3],
    sr_ratios=[8, 4, 2, 1],
):
        super().__init__()

        if in_chans is None:
            raise ValueError("in_chans should not be None")

        self.num_classes = num_classes
        self.depths = depths
        self.in_chans = in_chans

        self.debug_info = {}

        # RGB branch
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4,
            in_chans=self.in_chans[0], embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )

        # DSM / extra branch
        self.extra_patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4,
            in_chans=self.in_chans[1], embed_dim=embed_dims[0]
        )
        self.extra_patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.extra_patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.extra_patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )

        # CAF-Net modules
        self.acfm4 = AdaptiveCrossFrequencyModule(channels=embed_dims[3], low_radius=0.35)

        self.cmsg1 = CrossModalStructureGuidance(embed_dims[0])
        self.cmsg2 = CrossModalStructureGuidance(embed_dims[1])
        self.cmsg3 = CrossModalStructureGuidance(embed_dims[2])
        self.cmsg4 = CrossModalStructureGuidance(embed_dims[3])

        self.uaf1 = UncertaintyAwareFusion(embed_dims[0])
        self.uaf2 = UncertaintyAwareFusion(embed_dims[1])
        self.uaf3 = UncertaintyAwareFusion(embed_dims[2])
        self.uaf4 = UncertaintyAwareFusion(embed_dims[3])

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[0]
            )
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[0]
            )
            for i in range(depths[0])
        ])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[1]
            )
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[1]
            )
            for i in range(depths[1])
        ])
        self.extra_norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[2]
            )
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[2]
            )
            for i in range(depths[2])
        ])
        self.extra_norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[3]
            )
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[3]
            )
            for i in range(depths[3])
        ])
        self.extra_norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
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
            raise TypeError("pretrained must be a str or None")

    def _uaf_forward(self, uaf_module, rgb_feat, dsm_feat):
        out = uaf_module(rgb_feat, dsm_feat)
        if isinstance(out, (tuple, list)):
            if len(out) == 3:
                fused, w_rgb, w_dsm = out
            elif len(out) == 2:
                fused, w_rgb = out
                w_dsm = None
            else:
                fused = out[0]
                w_rgb, w_dsm = None, None
        else:
            fused = out
            w_rgb, w_dsm = None, None
        return fused, w_rgb, w_dsm

    def fusion_loss(self, f_cmsg, f_acfm, f_uaf, rgb_feat, dsm_feat):
        loss_acfm_uaf = F.l1_loss(f_acfm, f_uaf)
        loss_cmsg_uaf = F.l1_loss(f_cmsg, f_uaf)
        L_cons = loss_acfm_uaf + loss_cmsg_uaf

        _, _, H, W = rgb_feat.shape
        low_mask = self.acfm4._build_low_mask(H, W, rgb_feat.device, rgb_feat.dtype)
        low_freq_rgb = rgb_feat * low_mask
        low_freq_dsm = dsm_feat * low_mask
        low_L_cons = F.mse_loss(low_freq_rgb, low_freq_dsm)

        return L_cons, low_L_cons

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
        f1, _, _ = self._uaf_forward(self.uaf1, x_rgb, x_e)
        outs_semantic.append(f1)

        # Stage 2
        x_rgb, H, W = self.patch_embed2(f1)
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
        f2, _, _ = self._uaf_forward(self.uaf2, x_rgb, x_e)
        outs_semantic.append(f2)

        # Stage 3
        x_rgb, H, W = self.patch_embed3(f2)
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
        f3, _, _ = self._uaf_forward(self.uaf3, x_rgb, x_e)
        outs_semantic.append(f3)

        # Stage 4
        x_rgb, H, W = self.patch_embed4(f3)
        x_e, _, _ = self.extra_patch_embed4(x_e)
        for blk in self.block4:
            x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block4:
            x_e = blk(x_e, H, W)

        x_rgb = self.norm4(x_rgb)
        x_e = self.extra_norm4(x_e)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        rgb_stage4 = x_rgb
        dsm_stage4 = x_e

        f_cmsg = self.cmsg4(rgb_stage4, dsm_stage4)
        f_acfm = self.acfm4(rgb_stage4, dsm_stage4)
        f_uaf, w_rgb, w_dsm = self._uaf_forward(self.uaf4, f_cmsg, dsm_stage4)

        outs_semantic.append(f_uaf)

        self.debug_info = {
            "rgb_stage4": rgb_stage4.detach(),
            "dsm_stage4": dsm_stage4.detach(),
            "f_cmsg": f_cmsg.detach(),
            "f_acfm": f_acfm.detach(),
            "f_uaf": f_uaf.detach(),
            "w_rgb": None if w_rgb is None else w_rgb.detach(),
            "w_dsm": None if w_dsm is None else w_dsm.detach(),
        }

        L_cons, low_L_cons = self.fusion_loss(
            f_cmsg=f_cmsg,
            f_acfm=f_acfm,
            f_uaf=f_uaf,
            rgb_feat=rgb_stage4,
            dsm_feat=dsm_stage4,
        )

        return outs_semantic, L_cons.view(1), low_L_cons.view(1)

    def forward(self, x_rgb, x_e):
        return self.forward_features(x_rgb, x_e)


def load_dualpath_model(model, model_file, in_chans):
    t0 = time.time()

    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device("cpu"))
        if isinstance(raw_state_dict, dict) and "model" in raw_state_dict:
            raw_state_dict = raw_state_dict["model"]
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if "patch_embed" in k:
            if "patch_embed1.proj.weight" in k:
                rgb_v = _adapt_first_conv(v, in_chans[0])
                extra_v = _adapt_first_conv(v, in_chans[1])
                state_dict[k] = rgb_v
                state_dict[k.replace("patch_embed1", "extra_patch_embed1")] = extra_v
            else:
                state_dict[k] = v
                state_dict[k.replace("patch_embed", "extra_patch_embed")] = v
        elif "block" in k:
            state_dict[k] = v
            state_dict[k.replace("block", "extra_block")] = v
        elif "norm" in k:
            state_dict[k] = v
            state_dict[k.replace("norm", "extra_norm")] = v

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
    def __init__(self, in_chans, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=in_chans, **kwargs
        )


class mit_b1(RGBXTransformer):
    def __init__(self, in_chans, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=in_chans, **kwargs
        )


class mit_b2(RGBXTransformer):
    def __init__(self, in_chans, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=in_chans, **kwargs
        )


class mit_b3(RGBXTransformer):
    def __init__(self, in_chans, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=in_chans, **kwargs
        )


class mit_b4(RGBXTransformer):
    def __init__(self, in_chans, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=in_chans, **kwargs
        )


class mit_b5(RGBXTransformer):
    def __init__(self, in_chans, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=in_chans, **kwargs
        )