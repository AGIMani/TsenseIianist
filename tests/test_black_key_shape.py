"""Real compiled tapered surfaces, fixed reference centers and contact geometry.

Run on volc-a32 CPU. No GPU, policy or training required.
"""
import importlib.util
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from dm_control import mjcf
from tsensei_v3_2_pianist.models.piano import piano_mjcf, piano_constants as pc
from tsensei_v3_2_pianist.fingering_targets import key_reference, target_for_key


def body_vertices(model, geom):
    mesh = model.geom_dataid[geom]
    first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    return model.mesh_vert[first:first+count] @ rotation.reshape(3, 3).T + model.geom_pos[geom]


@pytest.mark.parametrize('actuated', [False, True])
def test_all_keys_shape_and_preserved_references(actuated):
    physics = mjcf.Physics.from_mjcf_model(piano_mjcf.build(add_actuators=actuated))
    m, d = physics.model.ptr, physics.data.ptr
    assert (m.njnt, m.ngeom, m.nsite) == (88, 89, 88)
    black = [i for i in range(88) if (i+21) % 12 in (1, 3, 6, 8, 10)]
    assert len(black) == 36
    for i in black:
        g = m.geom(f'black_key_geom_{i}').id
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        vertices = body_vertices(m, g)
        bottom = vertices[vertices[:, 2] < 0]
        top = vertices[vertices[:, 2] > 0]
        np.testing.assert_allclose(np.ptp(top, axis=0), [.090, .010, 0], atol=2e-8)
        np.testing.assert_allclose(np.ptp(bottom, axis=0), [.095, .012, 0], atol=2e-8)
        assert np.ptp(vertices[:, 2]) == pytest.approx(.018, abs=2e-8)
        assert top[:, 0].min() == pytest.approx(bottom[:, 0].min(), abs=2e-8)
        assert bottom[:, 0].max()-top[:, 0].max() == pytest.approx(.005, abs=2e-8)
        np.testing.assert_allclose((top.max(0)+top.min(0))[:2]/2, 0, atol=2e-8)
        sid, half_size = key_reference(m, g)
        np.testing.assert_allclose(half_size, [.045, .005, .009])
        np.testing.assert_allclose(m.body(int(m.geom_bodyid[g])).pos[[0, 2]], [-.030, .026])
        np.testing.assert_allclose(m.site_pos[sid], 0)
        target = target_for_key(m, d, g)
        np.testing.assert_allclose(target-d.site_xpos[sid], [.025, 0, .0045], atol=1e-12)
        # Full press must still use the body reference, never mesh principal axes.
        jid = int(m.body_jntadr[m.geom_bodyid[g]])
        d.qpos[m.jnt_qposadr[jid]] = m.jnt_range[jid, 1]
    mujoco.mj_forward(m, d)
    for i in black:
        g = m.geom(f'black_key_geom_{i}').id
        sid, _ = key_reference(m, g)
        np.testing.assert_allclose(target_for_key(m, d, g)-d.site_xpos[sid], [.025, 0, .0045], atol=1e-12)


def test_real_sloped_contact_surface():
    root = piano_mjcf.build()
    probe = root.worldbody.add('body', name='probe', mocap=True)
    probe.add('geom', name='probe_geom', type='sphere', size=[.00005])
    physics = mjcf.Physics.from_mjcf_model(root)
    m, d = physics.model.ptr, physics.data.ptr
    key = m.geom('black_key_geom_4').id
    sphere = m.geom('probe_geom').id
    center = m.body(m.geom_bodyid[key]).pos
    # Halfway up, half-width is 5.5 mm. An old 12 mm box would hit both.
    for y, collides in ((.0052, True), (.0058, False)):
        d.mocap_pos[0] = center + [0, y, 0]
        mujoco.mj_forward(m, d)
        distance = mujoco.mj_geomDistance(m, d, key, sphere, 1., None)
        assert (distance < 0) == collides
    # The extra 5 mm exists only at the foot, not across the upper surface.
    for z, collides in ((-.008, True), (.008, False)):
        d.mocap_pos[0] = center + [.048, 0, z]
        mujoco.mj_forward(m, d)
        assert (mujoco.mj_geomDistance(m, d, key, sphere, 1., None) < 0) == collides


def test_centers_dynamics_and_targets_match_verified_immediate_parent():
    parent = os.environ.get('PIANIST_SHAPE_PARENT')
    if not parent:
        pytest.skip('Set PIANIST_SHAPE_PARENT to verified immediate frozen source')
    modules = []
    for name in ('piano_constants', 'piano_mjcf'):
        path = Path(parent)/'tsensei_v3_2_pianist/models/piano'/f'{name}.py'
        spec = importlib.util.spec_from_file_location('shape_parent_'+name, path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        modules.append(module)
    modules[1].consts = modules[0]
    old = mjcf.Physics.from_mjcf_model(modules[1].build())
    new = mjcf.Physics.from_mjcf_model(piano_mjcf.build())
    for field in ('body_pos','body_mass','body_inertia','body_ipos','body_iquat',
                  'jnt_pos','jnt_axis','jnt_range','jnt_stiffness','qpos_spring',
                  'dof_damping','dof_armature','geom_friction','geom_solref',
                  'geom_solimp','geom_contype','geom_conaffinity'):
        np.testing.assert_allclose(getattr(new.model, field), getattr(old.model, field), atol=1e-14, rtol=0, err_msg=field)
    for i in range(88):
        color = 'black' if (i+21)%12 in (1,3,6,8,10) else 'white'
        g = new.model.ptr.geom(f'{color}_key_geom_{i}').id
        np.testing.assert_allclose(target_for_key(new.model.ptr,new.data.ptr,g),
            old.data.geom_xpos[g]+[old.model.geom_size[g,0]-(.020 if color=='black' else .040),0,.5*old.model.geom_size[g,2]])
    for _ in range(20):
        new.step()
    assert np.isfinite(new.data.qpos).all() and not np.any(new.data.warning.number)
