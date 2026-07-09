import os
import json
import logging
from typing import Dict, Any, List

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from skimage import io

from models.model_fsunet import Baseline
from utils import fix_random_seed, metrics, convert_from_color


logging.captureWarnings(True)
logger = logging.getLogger(__name__)


# =========================
# Model Zoo
# =========================
ROBUSTNESS_MODEL_ZOO = {
    "Baseline": {
        "ckpt": "/home/wsj/FDMF-Net-wsj/Baseline_Vaihingen_42/2026-03-31_10-04-22_baseline/results_Baseline_vaihingen/best_model_vaihingen",
        "use_cmsg": False,
        "use_acfm": False,
        "use_uaf": False,
    },
    "w/o UAF": {
        "ckpt": "/home/wsj/FDMF-Net-wsj/Baseline_Vaihingen_42/2026-03-31_10-05-35_without_UAF/results_Baseline_vaihingen/best_model_vaihingen",
        "use_cmsg": True,
        "use_acfm": True,
        "use_uaf": False,
    },
    "Full": {
        "ckpt": "/home/wsj/FDMF-Net-wsj/Baseline_Vaihingen_42/2026-03-31_10-06-24_Full/results_Baseline_vaihingen/best_model_vaihingen",
        "use_cmsg": True,
        "use_acfm": True,
        "use_uaf": True,
    }
}


# =========================
# Corruption Functions
# =========================
def apply_rgb_noise(img: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)


def apply_dsm_noise(dsm: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    noise = np.random.randn(*dsm.shape).astype(np.float32) * sigma
    return np.clip(dsm + noise, 0.0, 1.0).astype(np.float32)


def apply_dsm_shift(dsm: np.ndarray, max_shift: int = 4) -> np.ndarray:
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(dsm, shift=(dx, dy), axis=(0, 1)).astype(np.float32)


def apply_dsm_missing(dsm: np.ndarray) -> np.ndarray:
    return np.zeros_like(dsm, dtype=np.float32)


def apply_dsm_local_missing(
    dsm: np.ndarray,
    mask_ratio: float = 0.30,
    patch_size: int = 32
) -> np.ndarray:
    dsm = dsm.copy().astype(np.float32)
    h, w = dsm.shape[:2]

    if h < patch_size or w < patch_size:
        return dsm

    num_patches = int(mask_ratio * (h // patch_size) * (w // patch_size))
    num_patches = max(1, num_patches)

    for _ in range(num_patches):
        x = np.random.randint(0, h - patch_size + 1)
        y = np.random.randint(0, w - patch_size + 1)
        dsm[x:x + patch_size, y:y + patch_size] = 0.0

    return dsm


def apply_condition(img: np.ndarray, dsm: np.ndarray, condition: str):
    if condition == "RGB Noise":
        img = apply_rgb_noise(img, sigma=0.1)

    elif condition == "DSM Missing":
        dsm = apply_dsm_missing(dsm)

    elif condition == "DSM Noise":
        dsm = apply_dsm_noise(dsm, sigma=0.05)

    elif condition == "DSM Misalignment":
        dsm = apply_dsm_shift(dsm, max_shift=4)

    elif condition == "DSM Local Missing":
        dsm = apply_dsm_local_missing(dsm, mask_ratio=0.30, patch_size=32)

    return img.astype(np.float32), dsm.astype(np.float32)


# =========================
# Utils
# =========================
def build_model(cfg: DictConfig, dataset_cfg, model_meta: Dict[str, Any]):
    if cfg.training_dataset in ["Potsdam", "WHU"]:
        in_chans = [4, 1]
    elif cfg.training_dataset == "Vaihingen":
        in_chans = [3, 1]
    elif cfg.training_dataset == "YESeg":
        in_chans = [3, 3]
    else:
        in_chans = [3, 1]

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    # A full experiment checkpoint is loaded below, so avoid loading an unrelated
    # ImageNet backbone first.
    model_cfg.pretrained_backbone = None

    model = Baseline(
        cfg=model_cfg,
        num_classes=dataset_cfg.n_classes,
        in_chans=in_chans
    )

    model.backbone.use_cmsg = model_meta["use_cmsg"]
    model.backbone.use_acfm = model_meta["use_acfm"]
    model.backbone.use_uaf = model_meta["use_uaf"]

    # Disable one-off visualization side effects during robustness evaluation.
    model.vis_done = True
    model.backbone.vis_done = True

    model = model.cuda()
    return model


def smart_load_state_dict(model: nn.Module, ckpt_path: str):
    print(f"Loading checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    normalized_state = {}
    for k, v in state.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        normalized_state[new_k] = v

    try:
        model.load_state_dict(normalized_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint architecture does not match the evaluation model: {ckpt_path}\n{exc}"
        ) from exc

    print(f"Strict checkpoint load: OK ({len(normalized_state)} keys)")

    return model


def normalize_dsm(dsm: np.ndarray) -> np.ndarray:
    dsm = dsm.astype(np.float32)
    dsm_min = np.min(dsm)
    dsm_max = np.max(dsm)
    return ((dsm - dsm_min) / (dsm_max - dsm_min + 1e-8)).astype(np.float32)


def read_label(path: str, palette: Dict[Any, Any]) -> np.ndarray:
    lab = io.imread(path)

    if lab.ndim == 2:
        return lab.astype(np.int64)

    invert_palette = {tuple(v): k for k, v in palette.items()}
    return convert_from_color(lab, invert_palette).astype(np.int64)


def make_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]

    starts = list(range(0, length - tile + 1, stride))

    last = length - tile
    if starts[-1] != last:
        starts.append(last)

    return starts


def pad_to_tile_img(img: np.ndarray, tile: int):
    h, w = img.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)

    if pad_h == 0 and pad_w == 0:
        return img, h, w

    img_pad = np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect"
    )

    return img_pad, h, w


def pad_to_tile_dsm(dsm: np.ndarray, tile: int):
    h, w = dsm.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)

    if pad_h == 0 and pad_w == 0:
        return dsm, h, w

    dsm_pad = np.pad(
        dsm,
        ((0, pad_h), (0, pad_w)),
        mode="reflect"
    )

    return dsm_pad, h, w


# =========================
# Tile Inference
# =========================
def tile_inference(
    model: nn.Module,
    img: np.ndarray,
    dsm: np.ndarray,
    num_classes: int,
    tile: int = 256,
    stride: int = 192
) -> np.ndarray:

    model.eval()

    img, ori_h, ori_w = pad_to_tile_img(img, tile)
    dsm, _, _ = pad_to_tile_dsm(dsm, tile)

    h, w = img.shape[:2]

    prob_map = np.zeros((h, w, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w, 1), dtype=np.float32)

    x_starts = make_starts(h, tile, stride)
    y_starts = make_starts(w, tile, stride)

    with torch.inference_mode():
        for x in x_starts:
            for y in y_starts:
                img_patch = img[x:x + tile, y:y + tile, :]
                dsm_patch = dsm[x:x + tile, y:y + tile]

                img_t = torch.from_numpy(img_patch).permute(2, 0, 1).unsqueeze(0).cuda().float()
                dsm_t = torch.from_numpy(dsm_patch).unsqueeze(0).cuda().float()

                out, _, _ = model(img_t, dsm_t)

                if isinstance(out, (list, tuple)):
                    out = out[0]

                out_np = out.squeeze(0).detach().cpu().numpy()
                out_np = out_np.transpose(1, 2, 0)

                if out_np.shape[0] != tile or out_np.shape[1] != tile:
                    out_np = cv2.resize(
                        out_np,
                        (tile, tile),
                        interpolation=cv2.INTER_LINEAR
                    )

                prob_map[x:x + tile, y:y + tile, :] += out_np
                count_map[x:x + tile, y:y + tile, :] += 1.0

                del img_t, dsm_t, out

    prob_map = prob_map / np.maximum(count_map, 1.0)
    prob_map = prob_map[:ori_h, :ori_w, :]

    pred = np.argmax(prob_map, axis=-1).astype(np.int64)
    return pred


def extract_metrics(results: Dict[str, Any]):
    row = {
        "Kappa": results.get("Kappa", None),
        "OA": None,
        "mF1": None,
        "mIoU": None,
    }

    if "OA" in results and isinstance(results["OA"], dict):
        row["OA"] = results["OA"].get("total", None)

    if "F1" in results and isinstance(results["F1"], dict):
        row["mF1"] = results["F1"].get("mean", None)

    if "MIoU" in results and isinstance(results["MIoU"], dict):
        row["mIoU"] = results["MIoU"].get("mean", None)

    return row


# =========================
# Main
# =========================
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    fix_random_seed(cfg.seed)

    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg.training, resolve=True))

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.cuda_visible_devices))
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Training dataset:", cfg.training_dataset)

    dataset_cfg = cfg.dataset.datasets[cfg.training_dataset]

    conditions = [
        "Clean",
        "RGB Noise",
        "DSM Missing",
        "DSM Noise",
        "DSM Misalignment",
        "DSM Local Missing"
    ]

    tile_size = 256
    stride_size = 192

    save_root = os.path.join(os.getcwd(), f"robustness_engineering_{cfg.training_dataset.lower()}")
    os.makedirs(save_root, exist_ok=True)

    all_rows = []

    for model_name, meta in ROBUSTNESS_MODEL_ZOO.items():
        print("\n" + "=" * 100)
        print(f"Preparing model: {model_name}")

        model = build_model(cfg, dataset_cfg, meta)
        model = smart_load_state_dict(model, meta["ckpt"])
        print(
            "Ablation switches:",
            f"CMSG={model.backbone.use_cmsg},",
            f"ACFM={model.backbone.use_acfm},",
            f"UAF={model.backbone.use_uaf}"
        )
        model.eval()

        model_save_dir = os.path.join(
            save_root,
            model_name.replace("/", "_").replace(" ", "_")
        )
        os.makedirs(model_save_dir, exist_ok=True)

        for cond_idx, condition in enumerate(conditions):
            print("\n" + "-" * 80)
            print(f"Running: {model_name} | {condition}")

            fix_random_seed(cfg.seed + cond_idx)

            preds = []
            gts = []

            for image_idx, idx in enumerate(dataset_cfg.test_ids, start=1):
                print(f"[{model_name} | {condition}] Image {image_idx}/{len(dataset_cfg.test_ids)}: {idx}")

                img = io.imread(dataset_cfg.data_folder.format(idx)).astype(np.float32) / 255.0
                dsm = io.imread(dataset_cfg.dsm_folder.format(idx)).astype(np.float32)

                if "eroded_folder" in dataset_cfg:
                    gt_path = dataset_cfg.eroded_folder.format(idx)
                else:
                    gt_path = dataset_cfg.label_folder.format(idx)

                gt = read_label(gt_path, dataset_cfg.palette)
                dsm = normalize_dsm(dsm)

                img, dsm = apply_condition(img, dsm, condition)

                pred = tile_inference(
                    model=model,
                    img=img,
                    dsm=dsm,
                    num_classes=dataset_cfg.n_classes,
                    tile=tile_size,
                    stride=stride_size
                )

                if pred.shape != gt.shape:
                    pred = cv2.resize(
                        pred.astype(np.uint8),
                        (gt.shape[1], gt.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.int64)

                preds.append(pred)
                gts.append(gt)

                torch.cuda.empty_cache()

            results = metrics(
                np.concatenate([p.ravel() for p in preds]),
                np.concatenate([g.ravel() for g in gts]).ravel(),
                dataset_cfg.labels,
                dataset_cfg.n_classes
            )

            metric_row = extract_metrics(results)

            row = {
                "Model": model_name,
                "Condition": condition,
                **metric_row
            }

            all_rows.append(row)

            json_path = os.path.join(
                model_save_dir,
                f"{condition.replace(' ', '_')}.json"
            )

            json_safe = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    json_safe[k] = {
                        kk: float(vv) if isinstance(vv, (np.float32, np.float64, float, int)) else str(vv)
                        for kk, vv in v.items()
                    }
                else:
                    json_safe[k] = float(v) if isinstance(v, (np.float32, np.float64, float, int)) else str(v)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_safe, f, indent=2, ensure_ascii=False)

            print("Result:", row)
            print("Saved:", json_path)

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)

    csv_path = os.path.join(save_root, "robustness_all.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    pivot_path = os.path.join(save_root, "robustness_mIoU_pivot.csv")
    pivot = df.pivot(index="Model", columns="Condition", values="mIoU")
    pivot.to_csv(pivot_path, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print(df)

    print("\n" + "=" * 100)
    print("mIoU PIVOT")
    print(pivot)

    print("\nSaved CSV:", csv_path)
    print("Saved Pivot:", pivot_path)


if __name__ == "__main__":
    main()
