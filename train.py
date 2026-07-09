import logging
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage import io

from utils import (
    format_string, convert_from_color, count_sliding_window, grouper,
    sliding_window, CrossEntropy2d, dice_loss, metrics, convert_to_color
)

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


def add_noise_fn(x, sigma=0.1):
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0, 1)


def cosine_conflict(f1, f2, eps=1e-8):
    f1 = F.normalize(f1, p=2, dim=1, eps=eps)
    f2 = F.normalize(f2, p=2, dim=1, eps=eps)
    cos_sim = (f1 * f2).sum(dim=1)   # [B,H,W]
    conf_map = 1.0 - cos_sim
    return conf_map


def mean_conflict(conf_map, mask=None, eps=1e-8):
    if mask is None:
        return conf_map.mean().item()
    mask = mask.float()
    return ((conf_map * mask).sum() / (mask.sum() + eps)).item()


def get_boundary_mask_np(gt_patch):
    gt_patch = gt_patch.astype(np.uint8)
    edge = cv2.Canny(gt_patch, 0, 1)
    edge = (edge > 0).astype(np.float32)
    return edge


def get_backbone_debug_info(model):
    """
    Safe debug info getter.
    Fix: avoid AttributeError when backbone has no debug_info.
    """
    backbone = model.module.backbone if hasattr(model, "module") else model.backbone

    if hasattr(backbone, "debug_info"):
        return backbone.debug_info

    # ❗关键：没有 debug_info 就返回 None
    return None


def test(dataset_cfg, training_cfg, model, test_ids, all=False, test_loader=None,
         add_noise=False, noise_sigma=0.1, drop_dsm=False, collect_conflict=False):
    if dataset_cfg.name in ['Potsdam', 'Vaihingen']:
        stride = dataset_cfg.stride_size

    if model.training:
        model.eval()
    batch_size = training_cfg.batch_size
    window_size = tuple(training_cfg.window_size)
    n_classes = dataset_cfg.n_classes

    if dataset_cfg.name == 'Potsdam':
        test_images = (
            1 / 255 * np.asarray(io.imread(dataset_cfg.data_folder.format(id)), dtype='float32')
            for id in test_ids
        )
    elif dataset_cfg.name == 'Vaihingen':
        test_images = (
            1 / 255 * np.asarray(io.imread(dataset_cfg.data_folder.format(id)), dtype='float32')
            for id in test_ids
        )

    if dataset_cfg.name == 'Potsdam':
        dif_ids = [id for id in test_ids]
        test_dsms = (
            np.asarray(io.imread(dataset_cfg.dsm_folder.format(id)), dtype='float32')
            for id in dif_ids
        )
    else:
        test_dsms = (
            np.asarray(io.imread(dataset_cfg.dsm_folder.format(id)), dtype='float32')
            for id in test_ids
        )

    invert_palette = {tuple(v): k for k, v in dataset_cfg.palette.items()}
    test_labels = (
        convert_from_color(io.imread(dataset_cfg.label_folder.format(id)), invert_palette)
        for id in test_ids
    )

    if dataset_cfg.name in ['Potsdam', 'Vaihingen']:
        eroded_labels = (
            convert_from_color(io.imread(dataset_cfg.eroded_folder.format(id)), invert_palette)
            for id in test_ids
        )

    all_preds = []
    all_gts = []

    global_conf_list = []
    boundary_conf_list = []

    if dataset_cfg.name in ['Potsdam', 'Vaihingen']:
        with torch.inference_mode():
            total_images = len(test_ids)
            for image_idx, (img, dsm, gt, gt_e) in enumerate(
                zip(test_images, test_dsms, test_labels, eroded_labels),
                start=1
            ):
                image_start = time.perf_counter()
                pred = np.zeros(img.shape[:2] + (n_classes,), dtype=np.float32)

                dsm_min = np.min(dsm)
                dsm_max = np.max(dsm)
                dsm = (dsm - dsm_min) / (dsm_max - dsm_min + 1e-8)

                total_patches = count_sliding_window(
                    img,
                    step=stride,
                    window_size=window_size
                )
                total_batches = (total_patches + batch_size - 1) // batch_size
                progress_interval = max(1, total_batches // 20)
                logger.info(
                    "Image %d/%d: shape=%s, patches=%d, batches=%d, stride=%d",
                    image_idx,
                    total_images,
                    img.shape,
                    total_patches,
                    total_batches,
                    stride
                )

                patch_iter = grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))
                for batch_idx, coords in enumerate(patch_iter, start=1):
                    coords = [c for c in coords if c is not None]
                    if len(coords) == 0:
                        continue

                    image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = torch.from_numpy(image_patches).cuda()

                    if add_noise:
                        image_patches = add_noise_fn(image_patches, sigma=noise_sigma)

                    dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                    dsm_patches = np.asarray(dsm_patches)
                    dsm_patches = torch.from_numpy(dsm_patches).cuda()

                    if drop_dsm:
                        dsm_patches = torch.zeros_like(dsm_patches)

                    outs, _, _ = model(image_patches, dsm_patches)

                    if collect_conflict:
                        debug = get_backbone_debug_info(model)

                        if debug is None:
                            collect_conflict = False   # 🚨关键：直接关闭

                        if debug is not None:
                            f_cmsg = debug.get("f_cmsg")
                            f_acfm = debug.get("f_acfm")
                            if f_cmsg is not None and f_acfm is not None:
                                conf_map = cosine_conflict(f_cmsg, f_acfm)  # [B,Hc,Wc]
                                global_conf_list.append(mean_conflict(conf_map))

                                for bi, (x, y, w, h) in enumerate(coords):
                                    gt_patch = gt[x:x + w, y:y + h]
                                    boundary_np = get_boundary_mask_np(gt_patch)
                                    boundary_mask = torch.from_numpy(boundary_np).float().cuda()[None, None]
                                    boundary_mask = F.interpolate(
                                        boundary_mask,
                                        size=conf_map.shape[-2:],
                                        mode='nearest'
                                    ).squeeze(1)  # [1,Hc,Wc]

                                    boundary_conf = mean_conflict(conf_map[bi:bi + 1], boundary_mask)
                                    boundary_conf_list.append(boundary_conf)

                    outs = outs.data.cpu().numpy()

                    for out, (x, y, w, h) in zip(outs, coords):
                        out = out.transpose((1, 2, 0))
                        pred[x:x + w, y:y + h] += out

                    del outs

                pred = np.argmax(pred, axis=-1)
                all_preds.append(pred)
                all_gts.append(gt_e)
                logger.info(
                    "Image %d/%d finished in %.1fs",
                    image_idx,
                    total_images,
                    time.perf_counter() - image_start
                )

        results = metrics(
            np.concatenate([p.ravel() for p in all_preds]),
            np.concatenate([p.ravel() for p in all_gts]).ravel(),
            dataset_cfg.labels,
            dataset_cfg.n_classes
        )

        if collect_conflict:
            results["GlobalConflict"] = float(np.mean(global_conf_list)) if len(global_conf_list) > 0 else 0.0
            results["BoundaryConflict"] = float(np.mean(boundary_conf_list)) if len(boundary_conf_list) > 0 else 0.0

        if all:
            return results, all_preds, all_gts
        return results

    elif dataset_cfg.name in ['WHU', 'YESeg']:
        with torch.no_grad():
            for img, dsm, gt in test_loader:
                img, dsm = img.cuda(), dsm.cuda()

                if add_noise:
                    img = add_noise_fn(img, sigma=noise_sigma)
                if drop_dsm:
                    dsm = torch.zeros_like(dsm)

                outs, _, _ = model(img, dsm)
                outs = outs.data.cpu().numpy()

                pred = np.argmax(outs, axis=1)
                all_preds.append(pred)
                all_gts.append(gt)

        results = metrics(
            np.concatenate([p.ravel() for p in all_preds]),
            np.concatenate([p.ravel() for p in all_gts]).ravel(),
            dataset_cfg.labels,
            dataset_cfg.n_classes
        )

        if all:
            return results, all_preds, all_gts
        return results


def train(dataset_cfg, training_cfg, model, optimizer, scheduler, train_loader,
          weights, results_dir, test_loader=None):
    weights = weights.cuda()
    epochs = training_cfg.epochs
    save_epoch = training_cfg.save_epoch

    history = {
        'round': [],
        'train_loss': [],
        'Kappa': [],
        'OA_total': [],
        'MIoU_mean': [],
        'F1_mean': []
    }

    for label in dataset_cfg.labels:
        history[f'OA_{label}'] = []
        history[f'MIoU_{label}'] = []
        history[f'F1_{label}'] = []

    miou_best = 0.0
    epoch_best = -1

    for epoch in range(1, epochs + 1):
        logger.info('Train (epoch {}/{})'.format(epoch, epochs))
        model.train()
        batch_losses = []
        total_iter = len(train_loader)
        print_interval = max(1, total_iter // 10)

        for batch_idx, (opt, dsm, target) in enumerate(train_loader):
            opt, dsm, target = opt.cuda(), dsm.cuda(), target.cuda()
            optimizer.zero_grad()

            output, L_cons, low_L_cons = model(opt, dsm)
            loss_ce = CrossEntropy2d(output, target, weight=weights)
            loss_dice = dice_loss(output, target)

            # 裁剪 low_L_cons，防止负贡献项过大导致总 loss 为负
            max_low = (loss_ce + (L_cons * training_cfg.alpha) + (loss_dice * training_cfg.gamma)) / training_cfg.beta
            low_L_cons = torch.clamp(low_L_cons, max=max_low.detach())

            loss = (
                loss_ce
                + (L_cons * training_cfg.alpha)
                - (low_L_cons * training_cfg.beta)
                + (loss_dice * training_cfg.gamma)
            )

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            batch_losses.append(loss.item())
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_iter:
                print(f"Iter {batch_idx + 1}/{total_iter} | Loss: {loss.item():.4f}")

            del opt, dsm, target, loss

        epoch_loss = np.mean(batch_losses)

        if epoch % save_epoch == 0:
            model.eval()
            if dataset_cfg.name == 'WHU' and test_loader is not None:
                results_val = test(dataset_cfg, training_cfg, model, dataset_cfg.test_ids, all=False, test_loader=test_loader)
            elif dataset_cfg.name == 'YESeg' and test_loader is not None:
                results_val = test(dataset_cfg, training_cfg, model, dataset_cfg.test_ids, all=False, test_loader=test_loader)
            else:
                results_val = test(dataset_cfg, training_cfg, model, dataset_cfg.test_ids, all=False)
            model.train()

            miou = results_val['MIoU']['mean']

            history['round'].append(epoch)
            history['train_loss'].append(epoch_loss)
            history['Kappa'].append(results_val['Kappa'])

            history['OA_total'].append(results_val['OA']['total'])
            for i in dataset_cfg.labels:
                history[f'OA_{i}'].append(results_val['OA'][i])

            history['MIoU_mean'].append(results_val['MIoU']['mean'])
            for i in dataset_cfg.labels:
                history[f'MIoU_{i}'].append(results_val['MIoU'][i])

            history['F1_mean'].append(results_val['F1']['mean'])
            for i in dataset_cfg.labels:
                history[f'F1_{i}'].append(results_val['F1'][i])

            if miou > miou_best:
                if dataset_cfg.name == 'Vaihingen':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_vaihingen'))
                elif dataset_cfg.name == 'Potsdam':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_potsdam'))
                elif dataset_cfg.name == 'WHU':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_whu'))
                elif dataset_cfg.name == 'YESeg':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_yeseg'))

                miou_best = miou
                epoch_best = epoch

            logger.info('    Training Loss: {}'.format(epoch_loss))
            logger.info('    Kappa: {}'.format(results_val["Kappa"]))
            logger.info('    OA: {}'.format(results_val["OA"]))
            logger.info('    F1: {}'.format(results_val["F1"]))
            logger.info('    MIoU: {}'.format(results_val["MIoU"]))
            if "GlobalConflict" in results_val:
                logger.info('    GlobalConflict: {}'.format(results_val["GlobalConflict"]))
            if "BoundaryConflict" in results_val:
                logger.info('    BoundaryConflict: {}'.format(results_val["BoundaryConflict"]))
            logger.info("")
        else:
            history['round'].append(epoch)
            history['train_loss'].append(epoch_loss)
            history['Kappa'].append(0.0)

            history['OA_total'].append(0.0)
            for i in dataset_cfg.labels:
                history[f'OA_{i}'].append(0.0)

            history['MIoU_mean'].append(0.0)
            for i in dataset_cfg.labels:
                history[f'MIoU_{i}'].append(0.0)

            history['F1_mean'].append(0.0)
            for i in dataset_cfg.labels:
                history[f'F1_{i}'].append(0.0)

            logger.info('    Training Loss: {}'.format(epoch_loss))

    logger.info('Best epoch {}, MIoU best: {}'.format(epoch_best, miou_best))

    df = pd.DataFrame(history)
    if dataset_cfg.name == 'Vaihingen':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_vaihingen'))
    elif dataset_cfg.name == 'Potsdam':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_potsdam'))
    elif dataset_cfg.name == 'WHU':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_whu'))
    elif dataset_cfg.name == 'YESeg':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_yeseg'))

    logger.info('End of training !')


def visualize_testloader(model, test_loader, palette, save_root):
    model.cuda()
    model.eval()
    tile_idx = 0

    with torch.no_grad():
        for img, dsm, _ in test_loader:
            img, dsm = img.cuda(), dsm.cuda()
            pred, _, _ = model(img, dsm)
            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            for i in range(pred.shape[0]):
                color_pred = convert_to_color(pred[i], palette)
                io.imsave(
                    os.path.join(save_root, f"tile_{tile_idx}.png"),
                    color_pred,
                    check_contrast=False
                )
                tile_idx += 1
