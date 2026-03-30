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
# 1. 颜色映射与类别定义
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
N_CLASSES = 6 

ISPRS_COLORS = [
    (255, 255, 255), (0, 0, 255), (0, 255, 255), 
    (0, 255, 0), (255, 255, 0), (255, 0, 0)
]
CLASS_NAMES = ["Imp. surfaces", "Building", "Low veg.", "Tree", "Car", "Clutter"]
ISPRS_PALETTE = np.array(ISPRS_COLORS, dtype=np.uint8)

def colorize_mask(mask):
    mask = np.clip(mask, 0, N_CLASSES - 1)
    return ISPRS_PALETTE[mask]

# ==========================================
# 2. 拼接函数 (带图例)
# ==========================================
def create_comparison_row(img_list, titles, save_path):
    n = len(img_list)
    w, h = 256, 256
    gap = 15
    title_height, legend_height = 60, 60
    total_w = n * w + (n - 1) * gap
    total_h = h + title_height + legend_height + 20
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        legend_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
        legend_font = ImageFont.load_default()
    for i, (img_np, title) in enumerate(zip(img_list, titles)):
        img = Image.fromarray(img_np.astype(np.uint8))
        offset_x = i * (w + gap)
        canvas.paste(img, (offset_x, 10))
        tw = draw.textbbox((0, 0), title, font=font)[2] if hasattr(draw, 'textbbox') else draw.textsize(title, font=font)[0]
        draw.text((offset_x + (w - tw) // 2, h + 20), title, fill=(0, 0, 0), font=font)
    box_size, item_gap = 20, 30
    legend_items_w = []
    for name in CLASS_NAMES:
        nw = draw.textbbox((0, 0), name, font=legend_font)[2] if hasattr(draw, 'textbbox') else draw.textsize(name, font=legend_font)[0]
        legend_items_w.append(box_size + 8 + nw)
    total_legend_w = sum(legend_items_w) + item_gap * (len(CLASS_NAMES) - 1)
    current_lx = (total_w - total_legend_w) // 2
    ly = h + title_height + 20
    for i, (name, color) in enumerate(zip(CLASS_NAMES, ISPRS_COLORS)):
        draw.rectangle([current_lx, ly, current_lx + box_size, ly + box_size], fill=color, outline=(0,0,0))
        draw.text((current_lx + box_size + 8, ly - 2), name, fill=(0, 0, 0), font=legend_font)
        current_lx += legend_items_w[i] + item_gap
    canvas.save(save_path)

# ==========================================
# 3. 稳健的加载函数
# ==========================================
def load_robust_model(model_class, cfg, ckpt_path, device):
    print(f"\n>>> 正在加载模型权重: {ckpt_path}")
    # Potsdam 必须是 4通道 (RGB+IR)
    model = model_class(cfg=cfg.model, num_classes=N_CLASSES, in_chans=[4, 1]).to(device)
    
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        # 处理嵌套的字典
        if 'model' in state_dict: state_dict = state_dict['model']
        
        # 处理 DataParallel 前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # 【核心修改点】：使用 strict=False。即便层名对不上也不报错
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"    加载状态: 缺少 {len(msg.missing_keys)} 个层, 多余 {len(msg.unexpected_keys)} 个层")
    except Exception as e:
        print(f"    !!! 权重加载严重失败: {e}")
        
    model.eval()
    return model

# ==========================================
# 4. 推理主逻辑
# ==========================================
@hydra.main(config_path=".", config_name="config", version_base=None)
def run_visualization(cfg: DictConfig):
    save_dir = "./vis_potsdam_random_100" 
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.model_baseline import Baseline as BaselineClass
    from models.model import Baseline as FDMFClass
    from models.model_fsunet import Baseline as FSUNetClass

    data_root = '/data3/wsjdataset/OpenDataLab_Potsdam/raw/Potsdam/' 
    potsdam_id = '7_12' 
    rgb_path = os.path.join(data_root, '2_Ortho_RGB', f'top_potsdam_{potsdam_id}_RGB.tif')
    dsm_path = os.path.join(data_root, '1_DSM_normalisation', f'dsm_potsdam_{potsdam_id}_normalized_lastools.jpg')
    gt_path  = os.path.join(data_root, '5_Labels_for_participants', f'top_potsdam_{potsdam_id}_label.tif')

    model_configs = [
        {"name": "SegFormer", "class": BaselineClass, "path": "/home/wsj/FDMF-Net/Baseline_Potsdam_42-improved/2026-03-12_11-08-09_baseline/results_Baseline_potsdam/best_model_potsdam"},
        {"name": "FDMF-Net",  "class": FDMFClass,     "path": "/home/wsj/FDMF-Net/Baseline_Potsdam_42/2026-03-27_11-55-22/results_Baseline_potsdam/final_model_potsdam"},
        {"name": "FSU-Net",   "class": FSUNetClass,   "path": "/home/wsj/FDMF-Net/Baseline_Potsdam_42-improved/2026-03-14_10-24-14_创新点123/results_Baseline_potsdam/best_model_potsdam"}
    ]

    print(f">>> 读取 Potsdam Area {potsdam_id}...")
    full_rgb = io.imread(rgb_path)[:, :, :3] 
    full_dsm = io.imread(dsm_path)
    full_gt  = io.imread(gt_path) 
    H, W = full_rgb.shape[:2]

    # 随机 100 张
    all_coords = [(y, x) for y in range(0, H - 256, 256) for x in range(0, W - 256, 256)]
    random.seed(42)
    random.shuffle(all_coords)
    selected_coords = all_coords[:500]

    all_model_results = {}

    for m_cfg in model_configs:
        m_name = m_cfg["name"]
        model = load_robust_model(m_cfg["class"], cfg, m_cfg["path"], device)
        
        preds = []
        with torch.no_grad():
            for i, (y, x) in enumerate(selected_coords):
                rgb_p = full_rgb[y:y+256, x:x+256]
                dsm_p = full_dsm[y:y+256, x:x+256]
                
                r_t = torch.from_numpy(rgb_p.astype('float32')/255.).permute(2,0,1).unsqueeze(0).to(device)
                # 补齐第4通道
                r_t = torch.cat([r_t, r_t[:, :1, :, :]], dim=1) 
                
                d_min, d_max = dsm_p.min(), dsm_p.max()
                d_t = torch.from_numpy((dsm_p.astype('float32')-d_min)/(d_max-d_min+1e-6)).unsqueeze(0).unsqueeze(0).to(device)

                res = model(r_t, d_t)
                out = res[0] if isinstance(res, (list, tuple)) else res
                if isinstance(out, (list, tuple)): out = out[0]
                out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
                pred_idx = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
                preds.append(colorize_mask(pred_idx))
                if (i+1) % 50 == 0: print(f"  {m_name} 进度: {i+1}/100")

        all_model_results[m_name] = preds
        del model
        gc.collect()

    print("\n>>> 正在生成长图...")
    titles = ["RGB", "DSM", "GT", "SegFormer", "FDMF-Net", "FSU-Net"]
    for i, (y, x) in enumerate(selected_coords):
        rgb_p = full_rgb[y:y+256, x:x+256]
        dsm_p = full_dsm[y:y+256, x:x+256]
        gt_p  = full_gt[y:y+256, x:x+256]
        d_min, d_max = dsm_p.min(), dsm_p.max()
        d_vis = np.stack([((dsm_p-d_min)/(d_max-d_min+1e-6)*255).astype(np.uint8)]*3, axis=-1)
        
        row = [rgb_p, d_vis, gt_p, all_model_results["SegFormer"][i], all_model_results["FDMF-Net"][i], all_model_results["FSU-Net"][i]]
        create_comparison_row(row, titles, os.path.join(save_dir, f"Potsdam_{potsdam_id}_Y{y}_X{x}.png"))

    print(f"\n>>> 任务结束！")

if __name__ == "__main__":
    run_visualization()