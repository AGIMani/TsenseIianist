"""Score-relative future fingering, independent of a particular song or hand pose."""
import math

import torch


def future_fingering(fingers, control_dt, horizon):
    """Nearest future assignment for each currently idle finger, without wrapping.

    A horizon between score grid points includes only onsets actually within it.
    Returns assignment weights [T,10,88] and seconds until that assignment [T,10].
    """
    if not math.isfinite(horizon) or horizon < 0 or not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError('Anticipation horizon must be nonnegative and control_dt positive')
    if fingers.ndim != 3 or fingers.shape[1:] != (10, 88):
        raise ValueError('Fingering must have shape [time,10,88]')
    result = torch.zeros_like(fingers)
    seconds = torch.zeros(fingers.shape[:2], device=fingers.device, dtype=fingers.dtype)
    available = fingers.sum(-1) == 0
    steps = min(int(math.floor(horizon / control_dt + 1e-7)), len(fingers)-1)
    for offset in range(1, steps+1):
        use = available[:-offset] & (fingers[offset:].sum(-1) > 0)
        result[:-offset] = torch.where(use[..., None], fingers[offset:], result[:-offset])
        seconds[:-offset] = torch.where(use, offset*control_dt, seconds[:-offset])
        available[:-offset] &= ~use
    return result, seconds


def anticipation_fields(assignments, seconds, targets, fingertips, rest_top_z, clearance=.005):
    """A hover target above the unpressed key; no permission to press early."""
    if targets.ndim == 3:
        targets = targets[:, None].expand(-1, 10, -1, -1)
    weights = assignments / assignments.sum(-1, keepdim=True).clamp_min(1.)
    positions = (targets * weights[..., None]).sum(-2)
    positions[..., 2] = (weights * rest_top_z[None, None]).sum(-1) + clearance
    mask = assignments.sum(-1) > 0
    positions = torch.where(mask[..., None], positions, torch.zeros_like(positions))
    relative = torch.where(mask[..., None], positions-fingertips, torch.zeros_like(positions))
    # Finger-major: dx,dy,dz in metres; onset delay in seconds; active mask.
    observation = torch.cat((relative, seconds[..., None], mask[..., None].to(relative.dtype)), -1)
    return {'anticipatory_targets': positions, 'anticipatory_mask': mask,
            'anticipatory_time': seconds}, observation.flatten(1)
