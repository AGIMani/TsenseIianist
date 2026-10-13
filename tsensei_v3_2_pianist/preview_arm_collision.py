"""Explicit preview distance pairs for the user-selected forearm reward scope.

No solver forces: mocap arms are still kinematic. Convex mesh distances are
queried explicitly, including pairs disabled in the physical contact masks.
"""
import numpy as np
import mujoco
from .forearm_collision import is_forearm_collision_geom, is_palm_collision_geom


class ArmForearmCollision:
    def __init__(self, physics, arms):
        self.physics = physics
        m = physics.model.ptr
        self.groups = {side: {'arm': [], 'hand': []} for side in ('right', 'left')}
        self.excluded = []
        for side in self.groups:
            for i in range(m.ngeom):
                name = m.geom(i).name or ''
                if name.startswith(f'tj_collision_{side}_'):
                    self.groups[side]['arm'].append(i)
                elif name.startswith(f'{side}_/') and is_forearm_collision_geom(name):
                    self.groups[side]['hand'].append(i)
        self.pairs = []
        r,l = self.groups['right'],self.groups['left']
        self.pairs.extend((a,b) for a in r['hand'] for b in l['hand'])
        self.pairs.extend((a,b) for a in r['arm'] for b in l['arm'])
        for side, group in self.groups.items():
            for a in group['arm']:
                for other, target in self.groups.items():
                    for b in target['hand']:
                        # Only the own terminal Link7/palm mounting interface is excluded.
                        # Fingers touching Link7 remain unintended contacts.
                        if side == other and 'Link7_' in m.geom(a).name and is_palm_collision_geom(m.geom(b).name):
                            self.excluded.append((a,b))
                        else:
                            self.pairs.append((a,b))
        if any(len(g['hand']) != 21 or len(g['arm']) != 8 for g in self.groups.values()):
            raise ValueError('Expected 21 hand and 8 arm collision geoms per side')
        self.pairs = np.asarray(self.pairs, dtype=int).reshape(-1,2)
        self.margin = 1e-8

    def evaluate(self):
        m,d = self.physics.model.ptr,self.physics.data.ptr
        a,b = self.pairs.T
        # Conservative bounding-sphere rejection avoids most expensive mesh calls.
        radius = m.geom_rbound[a]+m.geom_rbound[b]+self.margin
        possible = np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b],axis=1) <= radius
        contacts=[]
        for first,second in self.pairs[possible]:
            distance = mujoco.mj_geomDistance(m,d,int(first),int(second),1e-4,None)
            if distance <= self.margin:
                contacts.append(dict(geom1=m.geom(first).name,geom2=m.geom(second).name,distance_m=float(distance)))
        return dict(forearm_reward=0. if contacts else .5,contacts=contacts,
                    pair_count=len(self.pairs), scope='cross_hands_cross_arms_and_arm_hand_v1',
                    detection='explicit_convex_distance_preview_only',physical_response=False)

    def manifest(self):
        m=self.physics.model.ptr
        return dict(groups={s:{k:[m.geom(i).name for i in v] for k,v in g.items()}
                            for s,g in self.groups.items()},
                    excluded_mount_pairs=[[m.geom(i).name for i in pair] for pair in self.excluded],
                    pairs=[[m.geom(i).name for i in pair] for pair in self.pairs],
                    result=self.evaluate())


def color_regions(model, collision):
    """Put selected collision meshes in visible group 4 without changing masks."""
    for side,group in collision.groups.items():
        for gid in group['arm']+group['hand']:
            model.geom_group[gid]=4
            model.geom_rgba[gid]=[1,.18,.12,.85] if side=='right' else [.12,.5,1,.85]
    for i in range(model.ngeom):
        if model.geom_group[i]==1:
            model.geom_rgba[i,3]=.22
