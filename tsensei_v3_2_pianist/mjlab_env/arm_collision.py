"""Detection-only collision tracking for the kinematic rigid-arm assembly.

The arm meshes remain collision-disabled in the force solver.  A separate
MuJoCo/Warp query model shares the main simulation transforms and samples
explicit arm-arm, arm-hand and arm-piano pairs after every physical substep.
"""
import mujoco
import numpy as np
import torch
import warp as wp

from .self_collision import mark_contacts


class ArmCollisionTracker:
    CATEGORIES = ('arm_arm', 'arm_hand', 'arm_piano')

    def __init__(self, model, pairs, *, num_envs, device):
        if not pairs:
            raise ValueError('Arm collision query requires at least one explicit pair')
        lookup = np.zeros((model.ngeom, model.ngeom), dtype=np.int32)
        names = []
        categories = []
        for index, (first, second, category) in enumerate(pairs, 1):
            if category not in self.CATEGORIES:
                raise ValueError(f'Unknown arm collision category: {category}')
            a = model.geom(first).id if isinstance(first, str) else int(first)
            b = model.geom(second).id if isinstance(second, str) else int(second)
            if a == b or lookup[a, b] != 0:
                raise ValueError('Duplicate or self arm collision pair')
            lookup[a, b] = lookup[b, a] = index
            names.append((model.geom(a).name or str(a), model.geom(b).name or str(b)))
            categories.append(category)
        self.pair_names = names
        self.pair_categories = categories
        self.lookup = wp.array(lookup, dtype=wp.int32, device=str(device))
        self.seen = torch.zeros((num_envs, len(pairs) + 1), dtype=torch.int32, device=device)
        self.pair_substep_seen = torch.zeros_like(self.seen)
        self.pair_substep_totals = torch.zeros_like(self.seen)
        # Per-control-step union used by rewards.  Pair totals remain
        # episode-level diagnostics and may count multiple pairs in one step.
        self.control_substep_any = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.seen_wp = wp.from_torch(self.seen)
        self.pair_substep_seen_wp = wp.from_torch(self.pair_substep_seen)
        self.category_indices = {
            category: torch.tensor([i + 1 for i, value in enumerate(categories)
                                    if value == category], dtype=torch.int64, device=device)
            for category in self.CATEGORIES
        }
        self.device = str(device)

    def clear(self, env_ids=None):
        if env_ids is None:
            self.seen.zero_()
            self.pair_substep_seen.zero_()
            self.pair_substep_totals.zero_()
            self.control_substep_any.zero_()
        else:
            self.seen[env_ids] = 0
            self.pair_substep_seen[env_ids] = 0
            self.pair_substep_totals[env_ids] = 0
            self.control_substep_any[env_ids] = 0

    def clear_step(self, env_ids=None):
        """Clear only the current control-step union used by rewards."""
        if env_ids is None:
            self.control_substep_any.zero_()
        else:
            self.control_substep_any[env_ids] = 0

    def sample(self, data, *, device=None):
        """Record unique pair occurrences for one 200Hz physical sample."""
        device = self.device if device is None else device
        self.pair_substep_seen.zero_()
        wp.launch(mark_contacts, dim=data.naconmax,
                  inputs=[data.nacon, data.contact.geom, data.contact.dist,
                          data.contact.worldid, self.lookup, self.pair_substep_seen_wp],
                  device=device)
        self.pair_substep_totals.add_(self.pair_substep_seen)
        self.control_substep_any.add_(self.pair_substep_seen[:, 1:].any(-1).to(torch.int32))
        # Keep endpoint diagnostics available without changing substep totals.
        self.seen.copy_(torch.maximum(self.seen, self.pair_substep_seen))

    def count(self, category=None):
        if category is None:
            indices = torch.arange(1, self.seen.shape[1], device=self.seen.device)
        else:
            if category not in self.category_indices:
                raise ValueError(f'Unknown arm collision category: {category}')
            indices = self.category_indices[category]
        return self.seen[:, indices].sum(-1)

    def count_substeps(self, category=None):
        if category is None:
            indices = torch.arange(1, self.seen.shape[1], device=self.seen.device)
        else:
            if category not in self.category_indices:
                raise ValueError(f'Unknown arm collision category: {category}')
            indices = self.category_indices[category]
        return self.pair_substep_totals[:, indices].sum(-1)

    def any(self):
        return self.count_substeps() > 0

    def count_control_substeps(self):
        return self.control_substep_any

    def contract(self):
        return dict(schema=1, mode='rigid_arm_detection_only_v1', sample_hz=200,
                    categories=list(self.CATEGORIES), pair_count=len(self.pair_names),
                    pairs=[dict(geom1=a, geom2=b, category=c)
                           for (a, b), c in zip(self.pair_names, self.pair_categories)],
                    force_solver='main scene excludes arm meshes; query has no reaction forces',
                    aggregation='unique_geom_pair_per_physics_substep_then_sum_per_control_step')
