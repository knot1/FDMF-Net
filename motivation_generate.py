# import os
# import gc
# import math
# import random
# from collections import OrderedDict

# import hydra
# import numpy as np
# import torch
# import torch.nn.functional as F
# from omegaconf import DictConfig
# from PIL import Image
# from skimage import io


# # =========================
# # 基础配置
# # =========================
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# N_CLASSES = 6
# PATCH_SIZE = 256
# NUM_SAMPLES = 50
# RANDOM_SEED = 42

# # 这里改成你的区域编号
# AREA_ID = "5"

# # 输出根目录
# SAVE_ROOT = "./motivation"

# # 数据根目录
# DATA_ROOT = "/data3/wsjdataset/Vaihingen_unzip/ISPRS_semantic_labeling_Vaihingen.zip/"

# # 模型权重路径
# SEGFORMER_CKPT = "/home/wsj/FDMF-Net/Baseline_Vaihingen_42-improved/2026-03-08_11-21-59_baseline_vaihingen/results_Baseline_vaihingen/best_model_vaihingen"
# CAFNET_CKPT = "/home/wsj/FDMF-Net/Baseline_Vaihingen_42/2026-03-31_10-06-24_Full/results_Baseline_vaihingen/best_model_vaihingen"


# # =========================
# # 颜色映射
# # =========================
# ISPRS_COLORS = [
#     (255, 255, 255),  # 0: Impervious surfaces
#     (0, 0, 255),      # 1: Building
#     (0, 255, 255),    # 2: Low vegetation
#     (0, 255, 0),      # 3: Tree
#     (255, 255, 0),    # 4: Car
#     (255, 0, 0),      # 5: Clutter/background
# ]
# ISPRS_PALETTE = np.array(ISPRS_COLORS, dtype=np.uint8)


# def colorize_mask(mask: np.ndarray) -> np.ndarray:
#     mask = np.clip(mask, 0, N_CLASSES - 1)
#     return ISPRS_PALETTE[mask]


# # =========================
# # 工具函数
# # =========================
# def ensure_dir(path: str):
#     os.makedirs(path, exist_ok=True)


# def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
#     arr = arr.astype(np.float32)
#     arr_min = arr.min()
#     arr_max = arr.max()
#     arr = (arr - arr_min) / (arr_max - arr_min + 1e-6)
#     arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
#     return arr


# def save_gray_map(arr_2d: np.ndarray, save_path: str):
#     img = normalize_to_uint8(arr_2d)
#     Image.fromarray(img).save(save_path)


# def save_rgb_image(arr: np.ndarray, save_path: str):
#     arr = np.clip(arr, 0, 255).astype(np.uint8)
#     Image.fromarray(arr).save(save_path)


# def save_dsm_image(dsm_patch: np.ndarray, save_path: str):
#     dsm_vis = normalize_to_uint8(dsm_patch)
#     Image.fromarray(dsm_vis).save(save_path)


# def tensor_from_rgb(rgb_patch: np.ndarray, device: torch.device) -> torch.Tensor:
#     # [H, W, 3] -> [1, 3, H, W]
#     return torch.from_numpy(rgb_patch.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


# def tensor_from_dsm(dsm_patch: np.ndarray, device: torch.device) -> torch.Tensor:
#     dsm_patch = dsm_patch.astype(np.float32)
#     dsm_min = dsm_patch.min()
#     dsm_max = dsm_patch.max()
#     dsm_patch = (dsm_patch - dsm_min) / (dsm_max - dsm_min + 1e-6)
#     return torch.from_numpy(dsm_patch).unsqueeze(0).unsqueeze(0).to(device)


# def load_clean_state_dict(ckpt_path: str, device: torch.device):
#     state_dict = torch.load(ckpt_path, map_location=device)
#     if isinstance(state_dict, dict) and "state_dict" in state_dict:
#         state_dict = state_dict["state_dict"]
#     if isinstance(state_dict, dict) and "model" in state_dict:
#         state_dict = state_dict["model"]

#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:] if k.startswith("module.") else k
#         new_state_dict[name] = v
#     return new_state_dict


# def find_backbone_with_debug_info(model):
#     """
#     在整个模型里找到有 debug_info 属性的 backbone。
#     你改完 encoder 后，RGBXTransformer 会有这个属性。
#     """
#     for _, m in model.named_modules():
#         if hasattr(m, "debug_info") and hasattr(m, "forward_features"):
#             return m
#     return None


# def get_prediction_output(model_output):
#     """
#     兼容不同 forward 返回形式：
#     - tensor
#     - tuple/list
#     - nested tuple/list
#     """
#     out = model_output
#     if isinstance(out, (list, tuple)):
#         out = out[0]
#     if isinstance(out, (list, tuple)):
#         out = out[0]
#     return out


# def compute_spatial_map_from_debug(debug_info: dict) -> np.ndarray:
#     """
#     Spatial Map: 用 rgb_stage4 的通道均值
#     """
#     feat = debug_info["rgb_stage4"]  # [1, C, H, W]
#     spatial_map = feat.mean(dim=1).squeeze(0).detach().cpu().numpy()  # [H, W]
#     return spatial_map


# def compute_frequency_map_from_debug(debug_info: dict) -> np.ndarray:
#     """
#     Frequency Map: 用 f_acfm 的通道均值
#     更贴合 CAF-Net 的 frequency 分支
#     """
#     feat = debug_info["f_acfm"]  # [1, C, H, W]
#     freq_map = feat.mean(dim=1).squeeze(0).detach().cpu().numpy()
#     return freq_map


# # 如果你想改成 FFT 版本，把上面的 frequency map 函数替换成这个
# def compute_frequency_map_fft_from_debug(debug_info: dict) -> np.ndarray:
#     feat = debug_info["rgb_stage4"]  # [1, C, H, W]
#     freq = torch.fft.fft2(feat)
#     freq = torch.abs(freq)
#     freq = torch.fft.fftshift(freq, dim=(-2, -1))
#     freq_map = freq.mean(dim=1).squeeze(0).detach().cpu().numpy()
#     return freq_map


# def sample_random_coordinates(H: int, W: int, patch_size: int, num_samples: int, seed: int = 42):
#     rng = random.Random(seed)
#     coords = []
#     for _ in range(num_samples):
#         y = rng.randint(0, H - patch_size)
#         x = rng.randint(0, W - patch_size)
#         coords.append((y, x))
#     return coords


# # =========================
# # 主逻辑
# # =========================
# @hydra.main(config_path=".", config_name="config", version_base=None)
# def main(cfg: DictConfig):
#     ensure_dir(SAVE_ROOT)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 1. 导入模型类
#     from models.model_baseline import Baseline as SegFormerClass
#     from models.model_cafnet import Baseline as CAFNetClass
#     import inspect
#     print("CAFNetClass file:", inspect.getfile(CAFNetClass))

#     # 2. 读取整幅图
#     rgb_path = os.path.join(DATA_ROOT, f"top/top_mosaic_09cm_area{AREA_ID}.tif")
#     dsm_path = os.path.join(DATA_ROOT, f"dsm/dsm_09cm_matching_area{AREA_ID}.tif")

#     print(f">>> 读取数据 Area {AREA_ID}")
#     full_rgb = io.imread(rgb_path)
#     full_dsm = io.imread(dsm_path)

#     if full_dsm.ndim == 3:
#         full_dsm = full_dsm[..., 0]

#     H, W = full_rgb.shape[:2]
#     print(f"RGB shape: {full_rgb.shape}, DSM shape: {full_dsm.shape}")

#     # 3. 随机采样 50 个窗口
#     coords = sample_random_coordinates(H, W, PATCH_SIZE, NUM_SAMPLES, RANDOM_SEED)

#     # 4. 加载模型
#     print(">>> 加载 SegFormer")
#     segformer = SegFormerClass(cfg=cfg.model, num_classes=N_CLASSES, in_chans=[3, 1]).to(device)
#     segformer.load_state_dict(load_clean_state_dict(SEGFORMER_CKPT, device))
#     segformer.eval()

#     print(">>> 加载 CAF-Net")
#     cafnet = CAFNetClass(cfg=cfg.model, num_classes=N_CLASSES, in_chans=[3, 1]).to(device)
#     print("cafnet class:", cafnet.__class__)
#     print("cafnet module:", cafnet.__class__.__module__)
#     cafnet.load_state_dict(load_clean_state_dict(CAFNET_CKPT, device))
#     cafnet.eval()

#     # 找 backbone
#     cafnet_backbone = find_backbone_with_debug_info(cafnet)
#     if cafnet_backbone is None:
#         raise RuntimeError(
#             "没有在 CAF-Net 模型中找到带 debug_info 的 backbone。"
#             "请先按我上面说的方式修改 encoder，把 rgb_stage4 / dsm_stage4 保存到 self.debug_info。"
#         )

#     print(">>> 开始生成 motivation 可视化")

#     with torch.no_grad():
#         for idx, (y, x) in enumerate(coords, start=1):
#             print(f"[{idx:02d}/{NUM_SAMPLES}] patch @ y={y}, x={x}")

#             patch_dir = os.path.join(SAVE_ROOT, f"y_{y}_x_{x}")
#             ensure_dir(patch_dir)

#             # 取 patch
#             rgb_patch = full_rgb[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
#             dsm_patch = full_dsm[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

#             # 输入张量
#             rgb_t = tensor_from_rgb(rgb_patch, device)
#             dsm_t = tensor_from_dsm(dsm_patch, device)

#             # SegFormer 推理
#             seg_out = segformer(rgb_t, dsm_t)
#             seg_out = get_prediction_output(seg_out)
#             seg_out = F.interpolate(seg_out, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)
#             seg_pred = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()
#             seg_vis = colorize_mask(seg_pred)

#             # CAF-Net 推理
#             caf_out = cafnet(rgb_t, dsm_t)
#             caf_out = get_prediction_output(caf_out)
#             caf_out = F.interpolate(caf_out, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)
#             caf_pred = torch.argmax(caf_out, dim=1).squeeze(0).cpu().numpy()
#             caf_vis = colorize_mask(caf_pred)

#             # 从 backbone.debug_info 里拿图
#             debug_info = cafnet_backbone.debug_info
#             if not debug_info:
#                 raise RuntimeError("CAF-Net backbone 的 debug_info 为空，说明中间特征没有被保存成功。")

#             spatial_map = compute_spatial_map_from_debug(debug_info)
#             frequency_map = compute_frequency_map_from_debug(debug_info)
#             # 如果你想用 FFT 版，改成这一句：
#             # frequency_map = compute_frequency_map_fft_from_debug(debug_info)

#             # 保存
#             save_rgb_image(rgb_patch, os.path.join(patch_dir, "RGB.png"))
#             save_dsm_image(dsm_patch, os.path.join(patch_dir, "DSM.png"))
#             save_gray_map(spatial_map, os.path.join(patch_dir, "Spatial_Map.png"))
#             save_gray_map(frequency_map, os.path.join(patch_dir, "Frequency_Map.png"))
#             save_rgb_image(seg_vis, os.path.join(patch_dir, "SegFormer.png"))
#             save_rgb_image(caf_vis, os.path.join(patch_dir, "CAF-Net.png"))

#             # 释放一点缓存
#             torch.cuda.empty_cache()

#     del segformer, cafnet
#     torch.cuda.empty_cache()
#     gc.collect()

#     print(f"\n>>> 完成，结果保存在: {SAVE_ROOT}")


# if __name__ == "__main__":
#     main()
import os
import gc
import random
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from skimage import io
import torchvision.transforms.functional as TF

# =========================
# 基础配置
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

N_CLASSES = 6
PATCH_SIZE = 256
NUM_SAMPLES = 50
RANDOM_SEED = 42
AREA_ID = "5"

SAVE_ROOT = "./motivation"
DATA_ROOT = "/data3/wsjdataset/Vaihingen_unzip/ISPRS_semantic_labeling_Vaihingen.zip/"

SEGFORMER_CKPT = "/home/wsj/FDMF-Net/Baseline_Vaihingen_42-improved/2026-03-08_11-21-59_baseline_vaihingen/results_Baseline_vaihingen/best_model_vaihingen"
CAFNET_CKPT = "/home/wsj/FDMF-Net/Baseline_Vaihingen_42/2026-03-31_10-06-24_Full/results_Baseline_vaihingen/best_model_vaihingen"


# =========================
# 颜色映射
# =========================
ISPRS_COLORS = [
    (255, 255, 255),  # 0: Impervious surfaces
    (0, 0, 255),      # 1: Building
    (0, 255, 255),    # 2: Low vegetation
    (0, 255, 0),      # 3: Tree
    (255, 255, 0),    # 4: Car
    (255, 0, 0),      # 5: Clutter/background
]
ISPRS_PALETTE = np.array(ISPRS_COLORS, dtype=np.uint8)


# =========================
# 工具函数
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, N_CLASSES - 1)
    return ISPRS_PALETTE[mask]


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr_min = arr.min()
    arr_max = arr.max()
    arr = (arr - arr_min) / (arr_max - arr_min + 1e-6)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def enhance_map_for_vis(arr: np.ndarray) -> np.ndarray:
    """
    对特征响应图做对比度增强，便于论文可视化
    """
    arr = arr.astype(np.float32)
    p2, p98 = np.percentile(arr, (2, 98))
    arr = np.clip((arr - p2) / (p98 - p2 + 1e-6), 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return arr


def array_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def array_to_pil_dsm(dsm_patch: np.ndarray) -> Image.Image:
    dsm_vis = normalize_to_uint8(dsm_patch)
    dsm_vis = np.stack([dsm_vis] * 3, axis=-1)
    return Image.fromarray(dsm_vis)


def array_to_pil_gray_map(arr_2d: np.ndarray) -> Image.Image:
    arr_vis = enhance_map_for_vis(arr_2d)
    arr_vis = np.stack([arr_vis] * 3, axis=-1)
    return Image.fromarray(arr_vis)


def tensor_from_rgb(rgb_patch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(
        rgb_patch.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_from_dsm(dsm_patch: np.ndarray, device: torch.device) -> torch.Tensor:
    dsm_patch = dsm_patch.astype(np.float32)
    dsm_min = dsm_patch.min()
    dsm_max = dsm_patch.max()
    dsm_patch = (dsm_patch - dsm_min) / (dsm_max - dsm_min + 1e-6)
    return torch.from_numpy(dsm_patch).unsqueeze(0).unsqueeze(0).to(device)


def load_clean_state_dict(ckpt_path: str, device: torch.device):
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def find_backbone_with_debug_info(model):
    for _, m in model.named_modules():
        if hasattr(m, "debug_info") and hasattr(m, "forward_features"):
            return m
    return None


def get_prediction_output(model_output):
    out = model_output
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out


def compute_spatial_map_from_debug(debug_info: dict, out_size):
    feat = debug_info["rgb_stage4"]   # [1, C, h, w]
    feat = feat.mean(dim=1, keepdim=True)  # [1,1,h,w]

    low = TF.gaussian_blur(feat, kernel_size=[11, 11])

    low = F.interpolate(
        low, size=out_size, mode="bilinear", align_corners=False
    )
    return low.squeeze().detach().cpu().numpy()


def compute_frequency_map_from_debug(debug_info: dict, out_size):
    feat = debug_info["rgb_stage4"]   # [1, C, h, w]
    feat = feat.mean(dim=1, keepdim=True)  # [1,1,h,w]

    low = TF.gaussian_blur(feat, kernel_size=[11, 11])
    high = feat - low

    high = F.interpolate(
        high, size=out_size, mode="bilinear", align_corners=False
    )
    return high.squeeze().detach().cpu().numpy()


def compute_frequency_map_fft_from_debug(debug_info: dict, out_size):
    feat = debug_info["rgb_stage4"]
    freq = torch.fft.fft2(feat)
    freq = torch.abs(freq)
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    freq_map = freq.mean(dim=1, keepdim=True)
    freq_map = F.interpolate(
        freq_map, size=out_size, mode="bilinear", align_corners=False
    )
    return freq_map.squeeze().detach().cpu().numpy()


def sample_random_coordinates(H: int, W: int, patch_size: int, num_samples: int, seed: int = 42):
    rng = random.Random(seed)
    coords = []
    for _ in range(num_samples):
        y = rng.randint(0, H - patch_size)
        x = rng.randint(0, W - patch_size)
        coords.append((y, x))
    return coords


def make_comparison_figure_2rows_from_images(images, titles, save_path, tile_size=256, gap=20, title_h=40, margin=20):
    """
    两排布局：
    第一排: RGB / DSM / GT
    第二排: Spatial / Frequency / SegFormer / CAF-Net
    """
    imgs = [img.resize((tile_size, tile_size)) for img in images]

    row1_imgs = imgs[:3]
    row1_titles = titles[:3]

    row2_imgs = imgs[3:]
    row2_titles = titles[3:]

    n1 = len(row1_imgs)
    n2 = len(row2_imgs)

    row1_w = n1 * tile_size + (n1 - 1) * gap
    row2_w = n2 * tile_size + (n2 - 1) * gap
    content_w = max(row1_w, row2_w)

    width = margin * 2 + content_w
    height = margin * 2 + (tile_size + title_h) * 2 + gap

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    # 第一排
    start_x1 = margin + (content_w - row1_w) // 2
    y1 = margin

    x = start_x1
    for img, title in zip(row1_imgs, row1_titles):
        canvas.paste(img, (x, y1))
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
        else:
            tw = draw.textsize(title, font=font)[0]
        tx = x + (tile_size - tw) // 2
        ty = y1 + tile_size + 8
        draw.text((tx, ty), title, fill=(0, 0, 0), font=font)
        x += tile_size + gap

    # 第二排
    start_x2 = margin + (content_w - row2_w) // 2
    y2 = margin + tile_size + title_h + gap

    x = start_x2
    for img, title in zip(row2_imgs, row2_titles):
        canvas.paste(img, (x, y2))
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
        else:
            tw = draw.textsize(title, font=font)[0]
        tx = x + (tile_size - tw) // 2
        ty = y2 + tile_size + 8
        draw.text((tx, ty), title, fill=(0, 0, 0), font=font)
        x += tile_size + gap

    canvas.save(save_path)


# =========================
# 主逻辑
# =========================
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    ensure_dir(SAVE_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.model_baseline import Baseline as SegFormerClass
    from models.model_cafnet import Baseline as CAFNetClass

    print("SEGFORMER_CKPT =", SEGFORMER_CKPT)
    print("CAFNET_CKPT    =", CAFNET_CKPT)

    rgb_path = os.path.join(DATA_ROOT, f"top/top_mosaic_09cm_area{AREA_ID}.tif")
    dsm_path = os.path.join(DATA_ROOT, f"dsm/dsm_09cm_matching_area{AREA_ID}.tif")
    gt_path = os.path.join(DATA_ROOT, f"gts_for_participants/top_mosaic_09cm_area{AREA_ID}.tif")

    print(f">>> 读取数据 Area {AREA_ID}")
    full_rgb = io.imread(rgb_path)
    full_dsm = io.imread(dsm_path)
    full_gt = io.imread(gt_path)

    if full_dsm.ndim == 3:
        full_dsm = full_dsm[..., 0]

    H, W = full_rgb.shape[:2]
    print(f"RGB shape: {full_rgb.shape}, DSM shape: {full_dsm.shape}")

    coords = sample_random_coordinates(H, W, PATCH_SIZE, NUM_SAMPLES, RANDOM_SEED)

    print(">>> 加载 SegFormer")
    segformer = SegFormerClass(cfg=cfg.model, num_classes=N_CLASSES, in_chans=[3, 1]).to(device)
    segformer.load_state_dict(load_clean_state_dict(SEGFORMER_CKPT, device), strict=False)
    segformer.eval()

    print(">>> 加载 CAF-Net")
    cafnet = CAFNetClass(cfg=cfg.model, num_classes=N_CLASSES, in_chans=[3, 1]).to(device)
    cafnet.load_state_dict(load_clean_state_dict(CAFNET_CKPT, device), strict=False)
    cafnet.eval()

    cafnet_backbone = find_backbone_with_debug_info(cafnet)
    if cafnet_backbone is None:
        raise RuntimeError("没有在 CAF-Net 模型中找到带 debug_info 的 backbone。")

    print(">>> 开始生成 motivation 可视化")

    with torch.no_grad():
        for idx, (y, x) in enumerate(coords, start=1):
            print(f"[{idx:02d}/{NUM_SAMPLES}] patch @ y={y}, x={x}")

            rgb_patch = full_rgb[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            dsm_patch = full_dsm[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            gt_patch = full_gt[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            rgb_t = tensor_from_rgb(rgb_patch, device)
            dsm_t = tensor_from_dsm(dsm_patch, device)

            # SegFormer
            seg_out = segformer(rgb_t, dsm_t)
            seg_out = get_prediction_output(seg_out)
            seg_out = F.interpolate(seg_out, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)
            seg_pred = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()
            seg_vis = colorize_mask(seg_pred)

            # CAF-Net
            caf_out = cafnet(rgb_t, dsm_t)
            caf_out = get_prediction_output(caf_out)
            caf_out = F.interpolate(caf_out, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)
            caf_pred = torch.argmax(caf_out, dim=1).squeeze(0).cpu().numpy()
            caf_vis = colorize_mask(caf_pred)

            debug_info = cafnet_backbone.debug_info
            if not debug_info:
                raise RuntimeError("CAF-Net backbone 的 debug_info 为空。")

            spatial_map = compute_spatial_map_from_debug(debug_info, (PATCH_SIZE, PATCH_SIZE))
            frequency_map = compute_frequency_map_from_debug(debug_info, (PATCH_SIZE, PATCH_SIZE))

            if gt_patch.ndim == 3:
                gt_vis = gt_patch
            else:
                gt_vis = colorize_mask(gt_patch)

            images = [
                array_to_pil_rgb(rgb_patch),
                array_to_pil_dsm(dsm_patch),
                array_to_pil_rgb(gt_vis),
                array_to_pil_gray_map(spatial_map),
                array_to_pil_gray_map(frequency_map),
                array_to_pil_rgb(seg_vis),
                array_to_pil_rgb(caf_vis),
            ]

            titles = [
                "RGB",
                "DSM",
                "GT",
                "Spatial Map",
                "Frequency Map",
                "SegFormer",
                "CAF-Net",
            ]

            save_name = f"y_{y}_x_{x}.png"
            save_path = os.path.join(SAVE_ROOT, save_name)

            make_comparison_figure_2rows_from_images(
                images=images,
                titles=titles,
                save_path=save_path,
                tile_size=256,
            )

            torch.cuda.empty_cache()

    del segformer, cafnet
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n>>> 完成，结果保存在: {SAVE_ROOT}")


if __name__ == "__main__":
    main()