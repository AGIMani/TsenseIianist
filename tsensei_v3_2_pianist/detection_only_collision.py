"""Inactive same-hand contacts: preserve detection without constraint forces.

MuJoCo >=3.11 uses detection threshold margin+gap and force threshold margin.
The negative force threshold exceeds the sum of geom bounding diameters, so
these finite convex hand geoms cannot reach it, even if fully overlapping.
"""
import mujoco
import numpy as np

MODE = 'same_hand_inactive_pairs_v1'


def pair_specs(model, *, include_cross_hand=False):
    if tuple(map(int, mujoco.__version__.split('.')[:2])) < (3, 11):
        raise ValueError('Detection-only contacts require MuJoCo >=3.11 margin/gap semantics')
    def side(i):
        name = model.body(int(model.geom_bodyid[i])).name
        return next((s for s in ('right_/', 'left_/') if name.startswith(s)), None)
    excluded = set(map(int, model.exclude_signature))
    explicit = {tuple(sorted((int(a), int(b)))) for a,b in zip(model.pair_geom1,model.pair_geom2)}
    result = []
    for a in range(model.ngeom):
        if not side(a):
            continue
        for b in range(a+1, model.ngeom):
            if not side(b) or (not include_cross_hand and side(a) != side(b)):
                continue
            ba,bb = sorted((int(model.geom_bodyid[a]),int(model.geom_bodyid[b])))
            wa,wb = int(model.body_weldid[ba]),int(model.body_weldid[bb])
            if (a,b) in explicit:
                raise ValueError('Existing explicit same-hand contact pair requires separate review')
            if not ((model.geom_contype[a] & model.geom_conaffinity[b]) or
                    (model.geom_contype[b] & model.geom_conaffinity[a])):
                continue
            if wa == wb or ((ba << 16) + bb) in excluded:
                continue
            if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)):
                if (int(model.body_weldid[model.body_parentid[wa]]) == wb or
                        int(model.body_weldid[model.body_parentid[wb]]) == wa):
                    continue
            radius = float(model.geom_rbound[a]+model.geom_rbound[b])
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError('Detection-only mode requires bounded hand collision geoms')
            gap = max(1.,4*radius)
            result.append(dict(name=f'self_detection_{a}_{b}',geom1=model.geom(a).name,
                               geom2=model.geom(b).name,margin=-gap,gap=gap,condim=1))
    return result


def apply_to_mjcf(root, *, include_cross_hand=False):
    """Preview-only model augmentation; source assets stay unchanged."""
    from dm_control import mjcf
    physics = mjcf.Physics.from_mjcf_model(root)
    specs = pair_specs(physics.model.ptr, include_cross_hand=include_cross_hand)
    for spec in specs:
        root.contact.add('pair', **spec)
    return len(specs)


def apply_to_xml(xml, model, *, include_cross_hand=False):
    import xml.etree.ElementTree as ET
    contact = xml.find('contact')
    if contact is None:
        contact = ET.SubElement(xml, 'contact')
    specs = pair_specs(model, include_cross_hand=include_cross_hand)
    for spec in specs:
        ET.SubElement(contact, 'pair', **{k:str(v) for k,v in spec.items()})
    return len(specs)


def disable_same_hand_response(model, *, include_cross_hand=False):
    """Partition collision masks; leave cross-hand, key and case contacts enabled."""
    if model.npair:
        raise ValueError('Detection-only GPU mode requires no explicit contact pairs')
    if not set(zip(map(int,model.geom_contype),map(int,model.geom_conaffinity))) <= {(0,0),(0,1),(1,1)}:
        raise ValueError('Unsupported collision masks; refusing to change unrelated contacts')
    for i in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[i])).name
        t, a = int(model.geom_contype[i]), int(model.geom_conaffinity[i])
        if name.startswith('right_/'):
            model.geom_contype[i], model.geom_conaffinity[i] = 2*t, (1 if include_cross_hand else 5)*a
        elif name.startswith('left_/'):
            model.geom_contype[i], model.geom_conaffinity[i] = 4*t, (1 if include_cross_hand else 3)*a
        else:
            model.geom_conaffinity[i] = 7*a
