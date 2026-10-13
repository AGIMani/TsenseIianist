"""Unique body contacts over all physics substeps of an action.

The legacy union counts each body pair once per control step. Optional
pair-substep totals count each pair once per physical substep, independent
of convex-mesh contact multiplicity. Endpoint refreshes never add a substep.
Native MuJoCo collision exclusions remain authoritative. The default is same-
hand only; explicit categories can also count opposite-hand and piano-case
contacts. Normal hand/key contacts remain excluded. This measures occurrence,
not contact force. Diagnostic pair mappings are selected by the user reward configuration.
Reward accumulation selects only different digits of the same hand; palm and
same-digit contacts remain physical and visible in diagnostics.

Additional categories are retained for contact diagnostics; the generic
collision_reward scoring term has been removed.
"""
import mujoco
import numpy as np
import torch
import warp as wp


def finger_identity(body_name):
    """Return (hand, digit) for the twenty URDF finger bodies per hand.

    Palm, mount and wrist bodies deliberately have no finger identity.
    All four thumb outputs belong to one digit, including both rotation links.
    """
    for side in ('left', 'right'):
        prefix = f'{side}_/{side}_hand_'
        if not body_name.startswith(prefix):
            continue
        link = body_name[len(prefix):]
        for digit in ('thumb', 'index', 'mid', 'ring', 'pinky'):
            suffixes = ('rota_block_link', 'rotaback_link2', 'MCP_link', 'PIP_link') if digit == 'thumb' else (
                'MCP_link', 'swing_link', 'PIP_link', 'DIP_link')
            if link in {f'{digit}_{suffix}' for suffix in suffixes}:
                return side, digit
    return None


def is_distinct_finger_pair(first, second):
    a, b = finger_identity(first), finger_identity(second)
    return a is not None and b is not None and a[0] == b[0] and a[1] != b[1]


@wp.kernel
def mark_contacts(nacon: wp.array(dtype=wp.int32),
                  geom: wp.array(dtype=wp.vec2i),
                  distance: wp.array(dtype=wp.float32),
                  world: wp.array(dtype=wp.int32),
                  pair_ids: wp.array2d(dtype=wp.int32),
                  seen: wp.array2d(dtype=wp.int32)):
    i = wp.tid()
    if i < nacon[0] and distance[i] <= 0.0:
        a, b = geom[i][0], geom[i][1]
        if a >= 0 and b >= 0:
            pair = pair_ids[a, b]
            if pair > 0:
                wp.atomic_max(seen, world[i], pair, 1)


@wp.kernel
def mark_diagnostic_contacts(nacon: wp.array(dtype=wp.int32),
                  geom: wp.array(dtype=wp.vec2i),
                  distance: wp.array(dtype=wp.float32),
                  world: wp.array(dtype=wp.int32),
                  pair_ids: wp.array2d(dtype=wp.int32),
                  categories: wp.array(dtype=wp.int32),
                  seen: wp.array2d(dtype=wp.int32),
                  present: wp.array2d(dtype=wp.int32)):
    i = wp.tid()
    if i < nacon[0] and distance[i] <= 0.0:
        a, b = geom[i][0], geom[i][1]
        if a >= 0 and b >= 0:
            pair = pair_ids[a, b]
            if pair > 0:
                wp.atomic_max(seen, world[i], pair, 1)
                category = categories[pair]
                if category >= 0:
                    wp.atomic_max(present, world[i], category, 1)


@wp.kernel
def accumulate_diagnostic_substep(present: wp.array2d(dtype=wp.int32),
                                 counts: wp.array2d(dtype=wp.int32)):
    world = wp.tid()
    own, other = present[world,0], present[world,1]
    counts[world,0] = counts[world,0] + own
    counts[world,1] = counts[world,1] + other
    counts[world,2] = counts[world,2] + wp.max(own,other)
    counts[world,3] = counts[world,3] + 1


def collision_weld_ids(model):
    """Distinguish independently moving mocap roots on older MuJoCo versions.

    Some versions assign all DOF-less bodies to world weld 0, even across two
    distinct mocaps. Such bodies can still have explicit collision pairs.
    Dynamic weld IDs remain exactly as compiled.
    """
    groups = np.asarray(model.body_weldid).copy()
    for body in np.flatnonzero(groups == 0):
        parent = int(body)
        while parent:
            if model.body_mocapid[parent] >= 0:
                groups[body] = -parent-1
                break
            parent = int(model.body_parentid[parent])
    return groups


class SelfCollisionTracker:
    def __init__(self, model, num_envs, device, *, categories=('self',), ignored_body_pairs=(),
                 diagnostics=False, accumulate_pair_substeps=False):
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ''
                 for i in range(model.nbody)]
        if set(categories)-{'self','inter_hand','piano_case'}:
            raise ValueError('Unknown collision category')
        if 'piano_case' in categories and 'piano/base' not in names:
            raise ValueError('piano_case mapping requires the named piano/base body')
        pairs = {}
        weld_ids = collision_weld_ids(model)
        self.pair_names = []
        self.pair_categories = []
        ignored = {tuple(sorted(p)) for p in ignored_body_pairs}
        for pair in ignored:
            if any(name not in names for name in pair):
                raise ValueError(f'Unknown ignored collision body pair: {pair}')
        def side(name):
            return 'right' if name.startswith('right_/') else 'left' if name.startswith('left_/') else None
        for a in range(model.nbody):
            for b in range(a + 1, model.nbody):
                first, second = side(names[a]), side(names[b])
                category = ('self' if first and first == second else
                            'inter_hand' if first and second else
                            'piano_case' if ((first and names[b] == 'piano/base')
                                             or (second and names[a] == 'piano/base')) else None)
                if (category in categories and weld_ids[a] != weld_ids[b]
                        and tuple(sorted((names[a], names[b]))) not in ignored):
                    pairs[a, b] = len(pairs) + 1
                    self.pair_names.append((names[a], names[b]))
                    self.pair_categories.append(category)
        lookup = np.zeros((model.ngeom, model.ngeom), dtype=np.int32)
        for a in range(model.ngeom):
            for b in range(a + 1, model.ngeom):
                key = tuple(sorted((int(model.geom_bodyid[a]), int(model.geom_bodyid[b]))))
                lookup[a, b] = lookup[b, a] = pairs.get(key, 0)
        self.lookup = wp.array(lookup, dtype=wp.int32, device=str(device))
        self.seen = torch.zeros((num_envs, len(pairs) + 1), dtype=torch.int32, device=device)
        self.seen_wp = wp.from_torch(self.seen)
        self.accumulate_pair_substeps = accumulate_pair_substeps
        if accumulate_pair_substeps:
            self.pair_substep_seen = torch.zeros_like(self.seen)
            self.pair_substep_seen_wp = wp.from_torch(self.pair_substep_seen)
            self.pair_substep_totals = torch.zeros_like(self.seen)
        self.category_indices = {category: torch.tensor(
            [i+1 for i,c in enumerate(self.pair_categories) if c==category], device=device, dtype=torch.int64)
            for category in ('self','inter_hand','piano_case')}
        self.finger_pair_indices = torch.tensor(
            [i + 1 for i, pair in enumerate(self.pair_names) if is_distinct_finger_pair(*pair)],
            device=device, dtype=torch.int64)
        self.diagnostics = diagnostics
        if diagnostics:
            category_ids=[-1]+[{'self':0,'inter_hand':1}.get(c,-1) for c in self.pair_categories]
            self.diagnostic_categories=wp.array(category_ids,dtype=wp.int32,device=str(device))
            self.substep_present=wp.zeros((num_envs,2),dtype=wp.int32,device=str(device))
            # self, inter-hand, either hand category, sampled physics steps.
            self.substep_counts=torch.zeros((num_envs,4),dtype=torch.int32,device=device)
            self.substep_counts_wp=wp.from_torch(self.substep_counts)

    def clear(self, env_ids=None):
        if env_ids is None:
            self.seen.zero_()
            if self.accumulate_pair_substeps:
                self.pair_substep_seen.zero_()
                self.pair_substep_totals.zero_()
            if self.diagnostics:
                self.substep_counts.zero_()
        else:
            self.seen[env_ids]=0
            if self.accumulate_pair_substeps:
                self.pair_substep_seen[env_ids]=0
                self.pair_substep_totals[env_ids]=0
            if self.diagnostics:
                self.substep_counts[env_ids]=0

    def sample(self, sim, *, data=None, count_substep=False):
        d = sim.wp_data if data is None else data
        if self.accumulate_pair_substeps and count_substep:
            self.pair_substep_seen.zero_()
            wp.launch(mark_contacts, dim=d.naconmax,
                      inputs=[d.nacon, d.contact.geom, d.contact.dist, d.contact.worldid,
                              self.lookup, self.pair_substep_seen_wp], device=sim.wp_device)
            self.pair_substep_totals.add_(self.pair_substep_seen)
        if self.diagnostics and count_substep:
            self.substep_present.zero_()
            wp.launch(mark_diagnostic_contacts,dim=d.naconmax,
                      inputs=[d.nacon,d.contact.geom,d.contact.dist,d.contact.worldid,
                              self.lookup,self.diagnostic_categories,self.seen_wp,self.substep_present],
                      device=sim.wp_device)
            wp.launch(accumulate_diagnostic_substep,dim=self.seen.shape[0],
                      inputs=[self.substep_present,self.substep_counts_wp],device=sim.wp_device)
        else:
            wp.launch(mark_contacts, dim=d.naconmax,
                      inputs=[d.nacon, d.contact.geom, d.contact.dist, d.contact.worldid,
                              self.lookup, self.seen_wp], device=sim.wp_device)

    def count(self, categories=None):
        if categories is None:
            return self.seen[:, 1:].sum(-1)
        indices = torch.cat([self.category_indices[c] for c in categories])
        return self.seen[:, indices].sum(-1)

    def count_pair_substeps(self, categories=('self',)):
        """Unfiltered contact diagnostics, never the self-collision reward input."""
        if not self.accumulate_pair_substeps:
            raise RuntimeError('Pair-substep accumulation was not enabled')
        indices = torch.cat([self.category_indices[c] for c in categories])
        return self.pair_substep_totals[:, indices].sum(-1)

    def count_finger_pair_substeps(self):
        """Reward count: same hand, different digits, unique body pair/substep."""
        if not self.accumulate_pair_substeps:
            raise RuntimeError('Pair-substep accumulation was not enabled')
        return self.pair_substep_totals[:, self.finger_pair_indices].sum(-1)
