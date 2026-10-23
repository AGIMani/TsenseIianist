"""On-device, every-substep acceptance metrics (optional, not training overhead)."""
import numpy as np
import torch
from ..fingering_targets import key_reference


class PhysicsAudit:
    def __init__(self, scene):
        self.scene = scene
        self.overflow = torch.zeros(scene.num_envs, dtype=torch.int32, device=scene.device)
        self.max_constraints = torch.zeros_like(self.overflow)
        self.max_contacts = torch.zeros(1, dtype=torch.int32, device=scene.device)
        self.max_penetration = torch.zeros(scene.num_envs, device=scene.device)
        self.max_key_excess = torch.zeros_like(self.max_penetration)
        self.slots = torch.arange(scene.sim.wp_data.naconmax, device=scene.device)
        m = scene.model
        names = [m.geom(i).name or '' for i in range(m.ngeom)]
        self.is_key = torch.tensor(['key_geom_' in n for n in names], device=scene.device)
        self.is_hand = torch.tensor([n.startswith(('right_/', 'left_/')) for n in names], device=scene.device)
        levers = []
        for j in scene.bundle.key_ids:
            geoms = np.flatnonzero(m.geom_bodyid == m.jnt_bodyid[j])
            g = next(g for g in geoms if 'key_geom_' in names[g])
            levers.append(2*key_reference(m, g)[1][0])
        self.levers = scene.tensor(levers)

    def __call__(self, scene):
        d = scene.data
        self.overflow.bitwise_or_(d.overflow[:])
        torch.maximum(self.max_constraints, d.nefc[:], out=self.max_constraints)
        torch.maximum(self.max_contacts, d.nacon[:], out=self.max_contacts)
        joints = d.qpos[:, scene.key_qadr]
        excess = ((joints-scene.key_limits[:, 1]).clamp_min(0)*self.levers).amax(dim=1)
        torch.maximum(self.max_key_excess, excess, out=self.max_key_excess)
        geom = d.contact.geom[:].long().clamp(0, scene.model.ngeom-1)
        valid = (self.slots < d.nacon[0]) & ((self.is_key[geom[:, 0]] & self.is_hand[geom[:, 1]])
                    | (self.is_key[geom[:, 1]] & self.is_hand[geom[:, 0]]))
        depth = torch.where(valid, (-d.contact.dist[:]).clamp_min(0), 0.)
        world = d.contact.worldid[:].long().clamp(0, scene.num_envs-1)
        self.max_penetration.scatter_reduce_(0, world, depth, reduce='amax', include_self=True)

    def report(self):
        s = self.scene
        return {'overflow_bits': self.overflow.cpu().tolist(),
                'max_constraints_per_world': self.max_constraints.cpu().tolist(),
                'max_total_contacts': int(self.max_contacts.item()),
                'contact_capacity_total': s.sim.wp_data.naconmax,
                'constraint_capacity_per_world': s.sim.wp_data.njmax,
                'max_hand_key_penetration_m': self.max_penetration.cpu().tolist(),
                'max_key_front_overtravel_m': self.max_key_excess.cpu().tolist()}
