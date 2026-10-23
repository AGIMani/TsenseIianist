"""Physical activation at full travel; continuous correct-key reward retains its own tolerance."""
import numpy as np

KEY_PRESS_TOLERANCE = 0.05
KEY_ACTIVATION_FRACTION = 1.0


def validate_fraction(fraction):
    if not np.isfinite(fraction) or float(fraction) != KEY_ACTIVATION_FRACTION:
        raise ValueError('Key activation requires normalized fraction 1.0; old angular thresholds are unsupported')
    return float(fraction)


def key_active(qpos, limits, fraction=KEY_ACTIVATION_FRACTION):
    validate_fraction(fraction)
    limits = np.asarray(limits)
    if limits.ndim != 2 or limits.shape[1] != 2 or not np.isfinite(limits).all() or np.any(limits[:, 0] != 0) or np.any(limits[:, 1] <= 0):
        raise ValueError('Key activation requires finite [0, positive] key limits')
    return np.clip(qpos, 0, limits[:, 1]) >= limits[:, 1] * fraction
