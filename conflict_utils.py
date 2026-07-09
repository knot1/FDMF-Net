import torch
import torch.nn.functional as F

def cosine_conflict(f1, f2):
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)
    sim = (f1 * f2).sum(dim=1)
    return 1 - sim   # 越小越一致

def mean_conflict(conf_map, mask=None):
    if mask is None:
        return conf_map.mean().item()
    return (conf_map * mask).sum().item() / (mask.sum().item() + 1e-6)