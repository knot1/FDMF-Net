import torch
import numpy as np
import random

# =========================
# robustness
# =========================
def noise(x, sigma=0.1):
    return torch.clamp(x + torch.randn_like(x) * sigma, 0, 1)

def shift(dsm, max_shift=8):
    b, h, w = dsm.shape
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    out = torch.zeros_like(dsm)

    x1 = max(0, dx)
    x2 = min(h, h + dx)
    y1 = max(0, dy)
    y2 = min(w, w + dy)

    out[:, x1:x2, y1:y2] = dsm[:, max(0, -dx):max(0, -dx)+(x2-x1),
                               max(0, -dy):max(0, -dy)+(y2-y1)]
    return out

def local_drop(dsm, ratio=0.3, patch=32):
    b, h, w = dsm.shape
    mask = torch.ones_like(dsm)

    for i in range(0, h, patch):
        for j in range(0, w, patch):
            if random.random() < ratio:
                mask[:, i:i+patch, j:j+patch] = 0

    return dsm * mask

def degrade(img, sigma=0.05):
    return torch.clamp(img + torch.randn_like(img) * sigma, 0, 1)


# =========================
# sliding window (stable)
# =========================
def sliding_window(model, img, dsm, num_classes, window=256, stride=128):

    _, H, W = img.shape

    pred = torch.zeros((num_classes, H, W)).cuda()
    count = torch.zeros((1, H, W)).cuda()

    for i in range(0, H - window + 1, stride):
        for j in range(0, W - window + 1, stride):

            img_p = img[:, i:i+window, j:j+window]
            dsm_p = dsm[:, i:i+window, j:j+window]

            out, _, _ = model(img_p.unsqueeze(0), dsm_p.unsqueeze(0))
            out = out.softmax(1).squeeze(0)

            pred[:, i:i+window, j:j+window] += out
            count[:, i:i+window, j:j+window] += 1

    pred = pred / (count + 1e-6)
    return pred.argmax(0)


# =========================
# MAIN TEST ENGINE
# =========================
def test(model,
         images,
         dsms,
         gts,
         dataset_cfg,
         condition="Clean"):

    model.eval()

    preds_all = []
    gts_all = []

    num_classes = dataset_cfg.n_classes

    with torch.no_grad():

        for img, dsm, gt in zip(images, dsms, gts):

            # -------------------------
            # numpy → tensor
            # -------------------------
            img = torch.from_numpy(img).float().cuda()
            dsm = torch.from_numpy(dsm).float().cuda()
            gt = torch.from_numpy(gt).long().cuda()

            # -------------------------
            # format fix
            # -------------------------
            if img.ndim == 3 and img.shape[-1] in [3,4]:
                img = img.permute(2, 0, 1)

            if dsm.ndim == 2:
                dsm = dsm.unsqueeze(0)

            # -------------------------
            # robustness
            # -------------------------
            if condition == "RGB Noise":
                img = noise(img)

            elif condition == "DSM Missing":
                dsm = torch.zeros_like(dsm)

            elif condition == "DSM Noise":
                dsm = noise(dsm)

            elif condition == "DSM Misalignment":
                dsm = shift(dsm)

            elif condition == "Local DSM Missing":
                dsm = local_drop(dsm)

            elif condition == "Optical Degradation":
                img = degrade(img)

            # -------------------------
            # inference
            # -------------------------
            pred = sliding_window(
                model,
                img,
                dsm,
                num_classes=num_classes
            )

            preds_all.append(pred.flatten().cpu())
            gts_all.append(gt.flatten().cpu())

    # -------------------------
    # metric (SAFE)
    # -------------------------
    preds_all = torch.cat(preds_all).numpy()
    gts_all = torch.cat(gts_all).numpy()

    acc = (preds_all == gts_all).mean()

    return {
        "MIoU": {"mean": float(acc)}
    }