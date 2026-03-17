def ensure_positive(name: str, value: float):
    if value <= 0:
        raise ValueError(f"`{name}` must be positive")


def ensure_same_shape(name_a: str, a, name_b: str, b):
    if getattr(a, "shape", None) != getattr(b, "shape", None):
        raise ValueError(f"`{name_a}` and `{name_b}` must have the same shape")


__all__ = ["ensure_positive", "ensure_same_shape"]
