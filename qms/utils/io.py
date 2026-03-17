from __future__ import annotations

import numpy as np


def save_npz_simulation(path: str, **arrays):
    """Save simulation arrays to a compressed NPZ file."""
    np.savez_compressed(path, **arrays)


__all__ = ["save_npz_simulation"]
