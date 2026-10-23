"""Score-relative future fingering, independent of a particular song or hand pose."""

import torch


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
