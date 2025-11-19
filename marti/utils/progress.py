# marti/utils/progress.py
import os
from contextlib import contextmanager

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

class _DummyBar:
    def __init__(self, *a, **k): pass
    def update(self, *a, **k): pass
    def set_postfix(self, *a, **k): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): pass

def is_progress_enabled() -> bool:
    # 只在 Driver 显示；给 Driver 进程设置环境变量 COMAS_PROGRESS=1
    return os.environ.get("COMAS_PROGRESS", "1") == "1" and tqdm is not None

def progress(total, desc="", enabled=None):
    if enabled is None:
        enabled = is_progress_enabled()
    if not enabled:
        return _DummyBar()
    return tqdm(
        total=total,
        desc=desc,
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.1,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
