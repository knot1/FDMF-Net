import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import io

from models.model import Baseline


# ===============================
# utils
# ===============================
def norm(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def to_gray(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim == 3:
        x = x.mean(0)
    return norm(x)


def to_rgb(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.shape[0] >= 3:
        x = x[:3]
        x = np.transpose(x, (1, 2, 0))
    return norm(x)


def resize_map(x, out_size=256, mode="bilinear"):
    """
    x: torch.Tensor or np.ndarray
    return: [out_size, out_size]
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            return cv2.resize(x, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
        elif x.ndim == 3:
            x = x.mean(axis=0)
            return cv2.resize(x, (out_size, out_size), interpolation=cv2.INTER_CUBIC)

    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif x.ndim == 3:
            x = x.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
        elif x.ndim == 4:
            x = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
            x = x[:1]

        x = F.interpolate(x, size=(out_size, out_size), mode=mode, align_corners=False)
        return x[0, 0].numpy()

    raise TypeError("Unsupported input type for resize_map")


def sliding_window(img, size=256, step=256):
    H, W = img.shape[:2]
    for x in range(0, H - size + 1, step):
        for y in range(0, W - size + 1, step):
            yield x, y, size, size


# ===============================
# hook catcher
# ===============================
class Catcher:
    def __init__(self):
        self.data = {}

    def cmsg(self, m, inp, out):
        rgb, dsm = inp
        self.data["cmsg_out"] = out.detach()

        blur = m.dsm_blur(dsm)
        structure = dsm - blur
        structure = m.structure_refine(structure)
        attn = m.attn(torch.cat([rgb, structure], dim=1))

        self.data["cmsg_attn"] = attn.detach()
        self.data["cmsg_structure"] = structure.detach()

    def uaf(self, m, inp, out):
        rgb, dsm = inp

        score_r = m.rgb_conf(rgb)
        score_d = m.dsm_conf(dsm)
        w = F.softmax(torch.cat([score_r, score_d], dim=1) / m.temperature, dim=1)

        self.data["uaf_w"] = w.detach()
        self.data["uaf_out"] = (w[:, 0:1] * rgb + w[:, 1:2] * dsm).detach()

    def acfm(self, m, inp, out):
        rgb, dsm = inp

        # 对齐前：模态差异
        before = torch.abs(rgb - dsm).mean(1, keepdim=True)

        # 对齐后：直接使用 output，不能再调用 m(rgb,dsm)
        after = torch.abs(out - rgb).mean(1, keepdim=True)

        self.data["acfm_before"] = before.detach()
        self.data["acfm_after"] = after.detach()


# ===============================
# draw figure
# ===============================
def draw(rgb_show, dsm_show, c_full, save_path):
    """
    3x5 mechanism visualization
    每个子图 256x256
    """
    rows, cols = 3, 5
    cell = 256

    fig = plt.figure(figsize=(cols * cell / 100, rows * cell / 100), dpi=100)
    gs = fig.add_gridspec(rows, cols)

    # ========= Row 1: CMSG =========
    # 第1列 RGB
    ax = plt.subplot(gs[0, 0])
    ax.imshow(rgb_show)
    ax.set_title("RGB", fontsize=12)
    ax.axis("off")

    # 第2列 DSM
    ax = plt.subplot(gs[0, 1])
    ax.imshow(dsm_show, cmap='gray')
    ax.set_title("DSM", fontsize=12)
    ax.axis("off")

    # 第3列 w/o CMSG（原图对照）
    ax = plt.subplot(gs[0, 2])
    ax.imshow(rgb_show)
    ax.set_title("w/o CMSG", fontsize=12)
    ax.axis("off")

    # 第4列 w/ CMSG
    ax = plt.subplot(gs[0, 3])
    ax.imshow(resize_map(to_gray(c_full.data["cmsg_out"][0]), 256), cmap='jet')
    ax.set_title("w/ CMSG", fontsize=12)
    ax.axis("off")

    # 第5列 Attention
    ax = plt.subplot(gs[0, 4])
    ax.imshow(resize_map(to_gray(c_full.data["cmsg_attn"][0]), 256), cmap='jet')
    ax.set_title("Attention", fontsize=12)
    ax.axis("off")

    # ========= Row 2: UAF =========
    ax = plt.subplot(gs[1, 0])
    ax.imshow(rgb_show)
    ax.axis("off")

    ax = plt.subplot(gs[1, 1])
    ax.imshow(dsm_show, cmap='gray')
    ax.axis("off")

    # w/o UAF（原图对照）
    ax = plt.subplot(gs[1, 2])
    ax.imshow(rgb_show)
    ax.set_title("w/o UAF", fontsize=12)
    ax.axis("off")

    # w/ UAF
    ax = plt.subplot(gs[1, 3])
    ax.imshow(resize_map(to_gray(c_full.data["uaf_out"][0]), 256), cmap='jet')
    ax.set_title("w/ UAF", fontsize=12)
    ax.axis("off")

    # Weight（RGB weight）
    ax = plt.subplot(gs[1, 4])
    ax.imshow(resize_map(to_gray(c_full.data["uaf_w"][0][0]), 256), cmap='jet')
    ax.set_title("Weight", fontsize=12)
    ax.axis("off")

    # ========= Row 3: ACFM =========
    before = resize_map(to_gray(c_full.data["acfm_before"][0]), 256)
    after = resize_map(to_gray(c_full.data["acfm_after"][0]), 256)
    diff = np.abs(after - before)

    ax = plt.subplot(gs[2, 0])
    ax.imshow(rgb_show)
    ax.axis("off")

    ax = plt.subplot(gs[2, 1])
    ax.imshow(dsm_show, cmap='gray')
    ax.axis("off")

    ax = plt.subplot(gs[2, 2])
    ax.imshow(before, cmap='magma')
    ax.set_title("w/o ACFM", fontsize=12)
    ax.axis("off")

    ax = plt.subplot(gs[2, 3])
    ax.imshow(after, cmap='magma')
    ax.set_title("w/ ACFM", fontsize=12)
    ax.axis("off")

    ax = plt.subplot(gs[2, 4])
    ax.imshow(diff, cmap='hot')
    ax.set_title("Diff", fontsize=12)
    ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


# ===============================
# main run
# ===============================
@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("./vis_module", exist_ok=True)

    # ===============================
    # 不改你的数据集路径
    # ===============================
    root = "/data3/wsjdataset/Vaihingen_unzip/ISPRS_semantic_labeling_Vaihingen.zip"

    data_folder = os.path.join(root, "top/top_mosaic_09cm_area{}.tif")
    dsm_folder = os.path.join(root, "dsm/dsm_09cm_matching_area{}.tif")

    test_ids = ['5', '21', '15', '30']

    # ===============================
    # model
    # ===============================
    model_full = Baseline(
        cfg=None,
        num_classes=6,
        use_cmsg=True,
        use_acfm=True,
        use_uaf=True
    )
    model_full.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model_full.eval().to(device)

    # 只用 full model 做 hook
    c_full = Catcher()
    model_full.backbone.cmsg4.register_forward_hook(c_full.cmsg)
    model_full.backbone.uaf4.register_forward_hook(c_full.uaf)
    model_full.backbone.acfm4.register_forward_hook(c_full.acfm)

    count = 0

    for img_id in test_ids:
        print(f"Processing image {img_id}")

        rgb = io.imread(data_folder.format(img_id)).astype(np.float32) / 255.0
        dsm = io.imread(dsm_folder.format(img_id)).astype(np.float32)

        if dsm.ndim == 3:
            dsm = dsm[..., 0]

        dsm = (dsm - dsm.min()) / (dsm.max() - dsm.min() + 1e-8)

        for x, y, w, h in sliding_window(rgb, size=256, step=256):
            rgb_patch = rgb[x:x + w, y:y + h]
            dsm_patch = dsm[x:x + w, y:y + h]

            rgb_t = torch.from_numpy(rgb_patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            dsm_t = torch.from_numpy(dsm_patch).unsqueeze(0).float().to(device)

            # forward
            model_full(rgb_t, dsm_t)

            rgb_show = to_rgb(rgb_t[0])
            dsm_show = to_gray(dsm_t[0])

            save_path = f"./vis_module/{img_id}_{x}_{y}.png"
            draw(rgb_show, dsm_show, c_full, save_path)

            print("Saved:", save_path)

            count += 1
            if count >= args.num:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/home/wsj/FDMF-Net/Baseline_Vaihingen_42-improved/2026-03-16_21-06-21_baseline_vaihingen_innovation123温度系数2.0(对比试验)/results_Baseline_vaihingen/best_model_vaihingen",
        type=str
    )
    parser.add_argument("--num", default=5, type=int, help="number of tiles to visualize")
    args = parser.parse_args()

    run(args)