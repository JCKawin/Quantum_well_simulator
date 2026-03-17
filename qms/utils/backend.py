from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


def get_array_module(use_gpu: bool = False):
    if use_gpu and cp is not None:
        return cp
    return np


def to_numpy(a: Any) -> np.ndarray:
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


__all__ = ["get_array_module", "to_numpy", "cp"]
