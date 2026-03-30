import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage import io
import hydra
from omegaconf import DictConfig

# ==========================================
# 1. 常量与颜色映射 (ISPRS 标准)
# ==========================================
N_CLASSES = 6 
ISPRS_PALETTE = np.array([
    [255, 255, 255], # 0: Impervious surfaces (白)
    [0, 0, 255],     # 1: Building (蓝)
    [0, 255, 255],   # 2: Low vegetation (浅蓝/青)
    [0, 255, 0],     # 3: Tree (绿)
    [255, 255, 0],   # 4: Car (黄)
    [255, 0, 0],     # 5: Clutter/background (红)
], dtype=np.uint8)

def colorize_mask(mask):
    """将类别索引图转换为 RGB"""
    mask = np.clip(mask, 0, N_CLASSES - 1)
    return ISPRS_PALETTE[mask]

# ==========================================
# 2. 核心可视化函数 (仅横向拼接)
# ==========================================
def create_simple_comparison(img_list, save_path):
    """
    img_list: [RGB_np, GT_np, Pred_np] 均为 (256, 256, 3)
    """
    n = len(img_list)
    w, h = 256, 256
    gap = 10  # 图片之间的间距
    
    # 画布尺寸：宽度 = 图片数 * 256 + 间距， 高度 = 256
    total_w = n * w + (n - 1) * gap
    total_h = h
    
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    
    for i, img_np in enumerate(img_list):
        img = Image.fromarray(img_np.astype(np.uint8))
        offset_x = i * (w + gap)
        canvas.paste(img, (offset_x, 0))
        
    canvas.save(save_path)

# ==========================================
# 3. 执行推理与可视化任务
# ==========================================
@hydra.main(config_path=".", config_name="config", version_base=None)
def run_visualization(cfg: DictConfig):
    # --- [1. 路径配置] ---
    save_dir = "vis_results_patches"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f">>> 当前 Hydra 运行目录: {os.getcwd()}")

    # 模型权重与数据使用绝对路径
    checkpoint_path = '/home/wsj/FDMF-Net/Baseline_Vaihingen_42-improved/2026-03-24_11-12-01温度系数0.5/results_Baseline_vaihingen/best_model_vaihingen'
    data_root = '/data3/wsjdataset/Vaihingen_unzip/ISPRS_semantic_labeling_Vaihingen.zip/'
    rgb_fmt = os.path.join(data_root, 'top/top_mosaic_09cm_area{}.tif')
    dsm_fmt = os.path.join(data_root, 'dsm/dsm_09cm_matching_area{}.tif')
    gt_fmt  = os.path.join(data_root, 'gts_for_participants/top_mosaic_09cm_area{}.tif')

    # --- [2. 自定义切片案例] ---
    # patch_center: (y, x) 大图中的像素位置
    test_cases = [
        {'id': '5',  'patch_center': (1200, 1500), 'tag': 'case1'},
        {'id': '15', 'patch_center': (800, 1200),  'tag': 'case2'},
        {'id': '30', 'patch_center': (2100, 800),  'tag': 'case3'},
    ]

    # --- [3. 加载模型] ---
    from models.model import Baseline 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Baseline(cfg=cfg.model, num_classes=N_CLASSES, in_chans=[3, 1]).to(device)
    
    print(f"正在加载模型权重: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # --- [4. 推理循环] ---
    with torch.no_grad():
        for case in test_cases:
            tid = case['id']
            cy, cx = case['patch_center']
            tag = case['tag']

            print(f"正在裁剪 Area {tid} @ 坐标 ({cy}, {cx})...")

            try:
                # 1. 加载全尺寸大图
                img_rgb_full = io.imread(rgb_fmt.format(tid))
                img_dsm_full = io.imread(dsm_fmt.format(tid))
                img_gt_full  = io.imread(gt_fmt.format(tid))

                # 2. 裁剪 256x256 Patch
                y_s, y_e = cy - 128, cy + 128
                x_s, x_e = cx - 128, cx + 128
                
                # 边界保护
                rgb_p = img_rgb_full[y_s:y_e, x_s:x_e]
                dsm_p = img_dsm_full[y_s:y_e, x_s:x_e]
                gt_p  = img_gt_full[y_s:y_e, x_s:x_e]

                # 3. 数据预处理
                r_t = torch.from_numpy(rgb_p.astype('float32')/255.).permute(2,0,1).unsqueeze(0).to(device)
                d_min, d_max = dsm_p.min(), dsm_p.max()
                d_t = torch.from_numpy((dsm_p.astype('float32')-d_min)/(d_max-d_min+1e-6)).unsqueeze(0).unsqueeze(0).to(device)

                # 4. 推理
                result = model(r_t, d_t)
                output = result[0] if isinstance(result, (list, tuple)) else result
                if isinstance(output, (list, tuple)): output = output[0]
                
                output = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
                pred_idx = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                # 5. 可视化转换
                gt_vis = gt_p if gt_p.ndim == 3 else colorize_mask(gt_p)
                pred_vis = colorize_mask(pred_idx)
                
                # 6. 保存拼接图
                save_name = f"Area_{tid}_{tag}_y{cy}_x{cx}.png"
                save_path = os.path.join(save_dir, save_name)
                
                # 传入列表：[RGB原图, 真值GT, 模型预测]
                create_simple_comparison([rgb_p, gt_vis, pred_vis], save_path)
                print(f"成功保存: {save_path}")

            except Exception as e:
                print(f"!!! 处理失败: {e}")

if __name__ == "__main__":
    run_visualization()