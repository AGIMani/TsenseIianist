import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import mujoco
from tsensei_v3_2_pianist.physics_settings import (load, validate, save, from_model,
                                                   shadow_comparison, SolimpController)
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.mjlab_env.paths import DEFAULT_SCENE, DEFAULT_TASK, resolve_bundle


class ContactSettingsTest(unittest.TestCase):
    def test_profile_and_compiled_contact(self):
        b=SceneBundle(DEFAULT_SCENE);m=b.model;s=load()
        self.assertEqual((b.manifest['control_dt'],m.opt.timestep),(.025,.0025))
        self.assertEqual((int(m.opt.integrator),m.opt.impratio),(0,10.))
        self.assertEqual(s['lookahead_seconds'],.5)
        self.assertEqual(m.geom('piano/white_key_geom_0').priority,0)
        np.testing.assert_allclose(m.geom('piano/white_key_geom_0').solref,[.005,1])
        np.testing.assert_allclose(m.joint('piano/white_joint_0').solimp,[.9,.95,.001,.5,2])
        for side in ('R','L'):
            np.testing.assert_allclose(m.actuator(side+'_Palm_Base_Tz').ctrlrange,[-.025,0])
        comparison = shadow_comparison(from_model(m, b.manifest['control_dt']))
        self.assertTrue(comparison['matches_shadow'], comparison['differences'])
        np.testing.assert_allclose(comparison['hand_key_contact']['mixed_solref'], [.005, 1])

    def test_slider_changes_contact_response_and_preserves_qpos(self):
        b=SceneBundle(DEFAULT_SCENE);m=b.model;d=b.reset_cpu()
        # A penetrating pose provides real hand-key contacts for parameter inspection.
        for side in ('R','L'):
            j=m.actuator_trnid[m.actuator(side+'_Palm_Base_Tz').id,0]
            d.qpos[m.jnt_qposadr[j]]=-.015
        mujoco.mj_forward(m,d)
        def contacts():
            return [c.solimp.copy() for c in d.contact if any('key_geom_' in (m.geom(int(g)).name or '') for g in c.geom)]
        before=contacts();self.assertTrue(before)
        q=d.qpos.copy();ctrl=d.ctrl.copy()
        c=SolimpController(m,load());v=c.settings['piano']['solimp'].copy();v[0]=.2
        c.set_values('piano',v);c.apply();mujoco.mj_forward(m,d)
        after=contacts();self.assertTrue(after)
        self.assertAlmostEqual(before[0][0],.7)
        self.assertAlmostEqual(after[0][0],.35)
        np.testing.assert_array_equal(q,d.qpos);np.testing.assert_array_equal(ctrl,d.ctrl)
        np.testing.assert_allclose(m.joint('piano/white_joint_0').solimp,load()['key_limit']['solimp'])
        c.restore_shadow();c.apply()
        np.testing.assert_allclose(m.geom('piano/white_key_geom_0').solimp,[.9,.95,.001,.5,2])
        np.testing.assert_allclose(m.geom('right_/right_hand_index_DIP_link_collision_0').solimp,
                                   [.5,.99,.0001,.5,2])

    def test_save_validation_backup_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'settings.json';original=save(load(),p)
            s=load(p);s['hand']['solimp'][0]=.3;save(s,p,expected=original)
            self.assertEqual(load(p)['hand']['solimp'][0],.3)
            self.assertEqual(len(list((p.parent/'contact_settings_backups').glob('*.json'))),1)
            with self.assertRaises(ValueError):save(s,p,expected=original)
        s=load();s['hand']['solimp'][0]=float('nan')
        with self.assertRaises(ValueError):validate(s)

    def test_new_task_and_legacy_lookup(self):
        import json
        from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
        from tsensei_v3_2_pianist.mjlab_env.paths import OUTPUTS
        task=json.loads((DEFAULT_TASK/'task.json').read_text())
        self.assertEqual(task['contract']['observation_size'],2070)
        self.assertEqual(task['contract']['environment']['n_steps_lookahead'],20)
        old=OUTPUTS/'scene'
        self.assertEqual(resolve_bundle(None,sha256(old/'manifest.json'),'manifest.json',DEFAULT_SCENE),old)

    def test_gpu_slider_updates_captured_arrays(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
        from tsensei_v3_2_pianist.physics_settings import from_model
        scene=PianoScene(DEFAULT_SCENE)
        scene.step(scene.bundle.normalize(scene.bundle.neutral_action)[None])
        c=SolimpController(scene.model,from_model(scene.model,.025),scene)
        ptr=scene.sim.wp_model.geom_solimp.ptr
        values=c.settings['piano']['solimp'].copy();values[0]=.2
        c.set_values('piano',values);c.apply()
        scene.step(scene.bundle.normalize(scene.bundle.neutral_action)[None])
        self.assertEqual(ptr,scene.sim.wp_model.geom_solimp.ptr)
        self.assertEqual(scene.decimation,10)
        np.testing.assert_allclose(scene.sim.wp_model.geom_solimp.numpy()[0,c.indices['piano'],0],.2)
        self.assertTrue(torch.isfinite(scene.data.qpos).all())


if __name__=='__main__':unittest.main()
