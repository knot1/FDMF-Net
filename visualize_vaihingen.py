import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gc
from PIL import Image, ImageDraw, ImageFont
from skimage import io
import hydra
from omegaconf import DictConfig

# ==========================================
# 1. 基础配置与颜色映射
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
N_CLASSES = 6 

# ISPRS 标准颜色定义
ISPRS_COLORS = [
    (255, 255, 255), # 0: Impervious surfaces
    (0, 0, 255),     # 1: Building
    (0, 255, 255),   # 2: Low vegetation
    (0, 255, 0),     # 3: Tree
    (255, 255, 0),   # 4: Car
    (255, 0, 0),     # 5: Clutter/background
]

CLASS_NAMES = [
    "Imp. surfaces", 
    "Building", 
    "Low veg.", 
    "Tree", 
    "Car", 
    "Clutter"
]

ISPRS_PALETTE = np.array(ISPRS_COLORS, dtype=np.uint8)

def colorize_mask(mask):
    mask = np.clip(mask, 0, N_CLASSES - 1)
    return ISPRS_PALETTE[mask]

# ==========================================
# 2. 增强版对比图拼接函数 (带标题 + 带彩色图例)
# ==========================================
def create_comparison_row(img_list, titles, save_path):
    """
    img_list: 图像列表
    titles: 图像下方标题 ['RGB', 'DSM', 'GT', ...]
    """
    n = len(img_list)
    w, h = 256, 256
    gap = 15
    title_height = 60  # 图片标题高度
    legend_height = 60 # 图例区域高度
    
    total_w = n * w + (n - 1) * gap
    total_h = h + title_height + legend_height + 20
    
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        legend_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
        legend_font = ImageFont.load_default()

    # 1. 绘制图片和图片标题
    for i, (img_np, title) in enumerate(zip(img_list, titles)):
        img = Image.fromarray(img_np.astype(np.uint8))
        offset_x = i * (w + gap)
        canvas.paste(img, (offset_x, 10))
        
        # 图片标题居中
        if hasattr(draw, 'textbbox'):
            tw = draw.textbbox((0, 0), title, font=font)[2]
        else:
            tw = draw.textsize(title, font=font)[0]
        tx = offset_x + (w - tw) // 2
        draw.text((tx, h + 20), title, fill=(0, 0, 0), font=font)

    # 2. 绘制彩色图例 (位于最下方)
    # 计算图例的总宽度以便整体居中
    box_size = 20
    item_gap = 30
    legend_items_w = []
    for name in CLASS_NAMES:
        if hasattr(draw, 'textbbox'):
            nw = draw.textbbox((0, 0), name, font=legend_font)[2]
        else:
            nw = draw.textsize(name, font=legend_font)[0]
        legend_items_w.append(box_size + 8 + nw)
    
    total_legend_w = sum(legend_items_w) + item_gap * (len(CLASS_NAMES) - 1)
    start_lx = (total_w - total_legend_w) // 2
    ly = h + title_height + 20
    
    current_lx = start_lx
    for i, (name, color) in enumerate(zip(CLASS_NAMES, ISPRS_COLORS)):
        # 画色块
        draw.rectangle([current_lx, ly, current_lx + box_size, ly + box_size], 
                       fill=color, outline=(0,0,0))
        # 画文字
        draw.text((current_lx + box_size + 8, ly - 2), name, fill=(0, 0, 0), font=legend_font)
        # 移动坐标
        current_lx += legend_items_w[i] + item_gap
        
    canvas.save(save_path)

# ==========================================
# 3. 推理主逻辑
# ==========================================
@hydra.main(config_path=".", config_name="config", version_base=None)
def run_visualization(cfg: DictConfig):
    save_dir = "./vis_results_legend_100" 
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.model_baseline import Baseline as BaselineClass
    from models.model import Baseline as FDMFClass
    from models.model_fsunet import Baseline as FSUNetClass

    data_root = '/data3/wsjdataset/Vaihingen_unzip/ISPRS_semantic_labeling_Vaihingen.zip/'
    # 注意：这里改成了 Area 30
    area_id = '5'
    rgb_path = os.path.join(data_root, f'top/top_mosaic_09cm_area{area_id}.tif')
    dsm_path = os.path.join(data_root, f'dsm/dsm_09cm_matching_area{area_id}.tif')
    gt_path  = os.path.join(data_root, f'gts_for_participants/top_mosaic_09cm_area{area_id}.tif')

    model_configs = [
        {"name": "SegFormer", "class": BaselineClass, "path": "/home/wsj/FDMF-Net/Baseline_Vaihingen_42-improved/2026-03-08_11-21-59_baseline_vaihingen/results_Baseline_vaihingen/best_model_vaihingen"},
        {"name": "FDMF-Net",  "class": FDMFClass,     "path": "/home/wsj/FDMF-Net/Baseline_Vaihingen_42-improved/2026-03-16_21-06-21_baseline_vaihingen_innovation123温度系数2.0(对比试验)/results_Baseline_vaihingen/best_model_vaihingen"},
        {"name": "FSU-Net",   "class": FSUNetClass,   "path": "/home/wsj/FDMF-Net/Baseline_Vaihingen_42-improved/2026-03-24_11-12-01温度系数0.5/results_Baseline_vaihingen/best_model_vaihingen"}
    ]

    print(f">>> 正在读取 Area {area_id} 数据...")
    full_rgb = io.imread(rgb_path)
    full_dsm = io.imread(dsm_path)
    full_gt  = io.imread(gt_path)
    H, W = full_rgb.shape[:2]

    random.seed(42)
    all_coords = [(y, x) for y in range(0, H-256, 256) for x in range(0, W-256, 256)]
    random.shuffle(all_coords)
    selected = all_coords[:100]

    all_model_results = {}

    for m_cfg in model_configs:
        m_name = m_cfg["name"]
        print(f"\n[模型推理] {m_name}")
        model = m_cfg["class"](cfg=cfg.model, num_classes=N_CLASSES, in_chans=[3, 1]).to(device)
        state_dict = torch.load(m_cfg["path"], map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()

        preds = []
        with torch.no_grad():
            for i, (y, x) in enumerate(selected):
                rgb_p = full_rgb[y:y+256, x:x+256]
                dsm_p = full_dsm[y:y+256, x:x+256]
                r_t = torch.from_numpy(rgb_p.astype('float32')/255.).permute(2,0,1).unsqueeze(0).to(device)
                d_min, d_max = dsm_p.min(), dsm_p.max()
                d_t = torch.from_numpy((dsm_p.astype('float32')-d_min)/(d_max-d_min+1e-6)).unsqueeze(0).unsqueeze(0).to(device)

                res = model(r_t, d_t)
                out = res[0] if isinstance(res, (list, tuple)) else res
                if isinstance(out, (list, tuple)): out = out[0]
                out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
                pred_idx = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
                preds.append(colorize_mask(pred_idx))

        all_model_results[m_name] = preds
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # --- 拼接与图例生成 ---
    print("\n>>> 正在生成最终海报...")
    final_titles = ["RGB", "DSM", "GT", "SegFormer", "FDMF-Net", "FSU-Net"]
    for i, (y, x) in enumerate(selected):
        rgb_p = full_rgb[y:y+256, x:x+256]
        dsm_p = full_dsm[y:y+256, x:x+256]
        gt_p  = full_gt[y:y+256, x:x+256]

        d_min, d_max = dsm_p.min(), dsm_p.max()
        d_vis = ((dsm_p - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
        d_vis = np.stack([d_vis]*3, axis=-1)
        gt_vis = gt_p if gt_p.ndim == 3 else colorize_mask(gt_p)
        
        row_images = [
            rgb_p, d_vis, gt_vis,
            all_model_results["SegFormer"][i],
            all_model_results["FDMF-Net"][i],
            all_model_results["FSU-Net"][i]
        ]

        save_filename = f"Area{area_id}_Y{y}_X{x}_Comparison.png"
        create_comparison_row(row_images, final_titles, os.path.join(save_dir, save_filename))

    print(f"\n>>> 任务完成！结果保存在: {save_dir}")

if __name__ == "__main__":
    run_visualization()