"""Bounded engineering fingering rewards, independent of song or joint names.

These opt-in rules are not undisclosed paper shaping. They reuse the existing
fingering reward's [0, 1] budget; no additional positive reward term is added.
"""
import torch

from .task_logic import gaussian_tolerance, fingering_distances


def staged_scores(snapshot, tolerance=.01):
    """Per assigned finger/key score: approach 30%, actual depression 70%.

The near-contact factor is a geometric proxy, not a force/contact sensor:
    the fingertip must be between 3 mm below and 2 mm above the actual moving
    key top plane for full press credit. The gate falls to zero at 5 mm below
    or 7 mm above the plane, so a finger under the key cannot earn press credit.
Thus a hovering finger with zero key travel can earn at most 0.3, even
inside the old 1 cm distance tolerance. Key travel is clipped to [0, 1].
"""
    targets = snapshot['key_targets']
    if targets.ndim == 3:
        targets = targets[:, None, :, :]
    elif targets.ndim != 4:
        raise ValueError('key_targets must have shape (N,88,3) or (N,10,88,3)')
    difference = snapshot['fingertips'][:, :, None, :] - targets
    approach = gaussian_tolerance(torch.linalg.vector_norm(difference, dim=-1),
                                  tolerance, tolerance*10)
    aligned = gaussian_tolerance(torch.linalg.vector_norm(difference[..., :2], dim=-1),
                                 tolerance, tolerance*10)
    if 'key_surface_height' not in snapshot:
        raise ValueError('Staged fingering requires snapshot field key_surface_height')
    surface = snapshot['key_surface_height']
    if surface.ndim == 2:
        surface = surface[:, None, :]
    height = snapshot['fingertips'][:, :, None, 2]-surface
    near_contact = ((.007-height)/.005).clamp(0., 1.)*((height+.005)/.002).clamp(0., 1.)
    travel = snapshot['key_state'][:, None, :].clamp(0., 1.)
    return .3*approach + .7*aligned*near_contact*travel


def fingering_reward(snapshot, fingers, *, staged=False, anticipatory=False,
                     current_priority=False, tolerance=.01):
    scores = (staged_scores(snapshot, tolerance) if staged else
              gaussian_tolerance(fingering_distances(snapshot), tolerance, tolerance*10))
    total = (scores*fingers).sum((1, 2))
    count = fingers.sum((1, 2))
    if anticipatory:
        for field in ('anticipatory_targets', 'anticipatory_mask'):
            if field not in snapshot:
                raise ValueError('Anticipatory fingering requires snapshot field '+field)
        mask = snapshot['anticipatory_mask'].bool() & ~fingers.bool().any(-1)
        distance = torch.linalg.vector_norm(
            snapshot['fingertips']-snapshot['anticipatory_targets'], dim=-1)
        # Future-only silence earns at most half a fingering point. Current
        # assignments and future approaches share a single averaging budget.
        future_score = .5*gaussian_tolerance(distance, .003, .05)
        if current_priority:
            # Keep the current assignment mean independent of how many idle
            # fingers are preparing. Future positioning can fill only 20% of
            # its remaining budget, with future_mean itself capped at 0.5.
            # Thus current credit never decreases and its derivative is >=0.9;
            # the full term remains <=1. At perfect current credit the future
            # gradient is intentionally zero. Silence retains its <=0.5 cue.
            current_mean = total/count.clamp_min(1)
            future_mean = (future_score*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            return torch.where(count > 0,
                current_mean+(1-current_mean)*.2*future_mean, future_mean)
        total = total+(future_score*mask).sum(-1)
        count = count+mask.sum(-1)
    return total/count.clamp_min(1)
