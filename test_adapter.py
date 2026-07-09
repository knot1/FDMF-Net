import inspect
from test import test as raw_test


def adaptive_test(*, dataset_cfg, model, test_ids=None,
                  test_loader=None,
                  add_noise=False,
                  drop_dsm=False,
                  dsm_noise=False,
                  dsm_shift=False,
                  dsm_local_missing=False):

    """
    🔥 自动适配不同版本 test() 的统一接口
    """

    sig = inspect.signature(raw_test)
    params = sig.parameters

    kwargs = {}

    # =========================
    # 1. common required args
    # =========================
    if "dataset_cfg" in params:
        kwargs["dataset_cfg"] = dataset_cfg

    if "model" in params:
        kwargs["model"] = model

    # =========================
    # 2. dataset split handling
    # =========================
    if "test_ids" in params and test_ids is not None:
        kwargs["test_ids"] = test_ids

    if "test_loader" in params and test_loader is not None:
        kwargs["test_loader"] = test_loader

    # =========================
    # 3. corruption flags (safe pass)
    # =========================
    for k in ["add_noise", "drop_dsm", "dsm_noise", "dsm_shift", "dsm_local_missing"]:
        if k in params:
            kwargs[k] = locals()[k]

    # =========================
    # 4. call
    # =========================
    return raw_test(**kwargs)