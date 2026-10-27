"""Preparation planning primitives: no scene/model construction on this host."""
import importlib
import unittest
import numpy as np

class FirstNotePreparationTests(unittest.TestCase):
    def setUp(self):
        self.module=importlib.import_module('tsensei_v3_2_pianist.mjlab_env.first_note_preparation')

    def test_each_hand_first_chord_uses_exact_times(self):
        events=np.array([[.15,.25,40,2],[.15,.4,44,3],[.4,.5,42,1],[.8,1.,20,6]])
        rows=self.module.first_notes(events)
        self.assertEqual(rows,[[.15,.25,40,2],[.15,.4,44,3],[.8,1.,20,6]])

    def test_one_hand_and_unsorted_input(self):
        self.assertEqual(self.module.first_notes([[1,2,44,1],[0,1,40,2]]),[[0.,1.,40,2]])

    def test_reject_missing_conflicting_fingering_and_empty(self):
        for events in ([],[[0,1,40,-1]],[[0,1,40,1],[0,1,44,1]],[[0,1,88,1]],[[float('nan'),1,40,1]]):
            with self.subTest(events=events),self.assertRaises(ValueError):
                self.module.first_notes(events)

    def test_departure_is_continuous_with_bounded_acceleration(self):
        t=np.linspace(0,.05,501)
        q=self.module.departure_targets(np.array([0.]),np.array([.008]),t,.05)[:,0]
        self.assertAlmostEqual(q[0],0.)
        self.assertAlmostEqual(q[-1],.008)
        self.assertLess(abs(np.diff(q)/.0001).max(),.33)
        self.assertLess(abs(np.diff(q,n=2)/.0001**2).max(),15.)
        self.assertLess(abs((q[1]-q[0])/.0001),.001)
        self.assertLess(abs((q[-1]-q[-2])/.0001),.001)

    def test_quintic_duration_obeys_rate_limits_and_stationary_endpoints(self):
        start=np.array([0.,.2]); end=np.array([.6,1.2]); v=np.array([.4,2.]); a=np.array([.5,4.])
        duration=self.module.motion_duration(end-start,v,a,minimum=.1,control_dt=.05)
        times=np.arange(round(duration/.001)+1)*.001
        q=self.module.quintic_targets(start,end,times,duration)
        np.testing.assert_allclose(q[0],start,atol=0,rtol=0)
        np.testing.assert_allclose(q[-1],end,atol=1e-12)
        self.assertTrue((abs(np.diff(q,axis=0)/.001).max(axis=0)<=v+1e-6).all())
        self.assertTrue((abs(np.diff(q,n=2,axis=0)/.001**2).max(axis=0)<=a+1e-6).all())
        self.assertAlmostEqual(duration/.05,round(duration/.05))

if __name__=='__main__':unittest.main()

class FirstNoteEntryTests(unittest.TestCase):
    def test_entry_contract_and_learning_budget(self):
        from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config
        base=['--no-arms','--disable-wrist-tz','--disable-wrist-yaw','--action-profile','direct_joint_v1',
              '--finger-coupling','--finger-coupling-angle-ranges','configs/finger_coupling/angle_ranges_10_80.json']
        plain=effective_config(parser().parse_args(base))
        prepared=effective_config(parser().parse_args(base+['--preparation-mode','first_note_v1',
            '--preparation-time','2','--preparation-lift','0']))
        self.assertEqual(prepared['preparation']['hover'],.012)
        self.assertEqual(prepared['preparation']['settle_time'],.5)
        for key in plain.keys()-{'preparation'}:
            self.assertEqual(plain[key],prepared[key],key)

    def test_preparation_rejects_support_teleport_and_bad_hover(self):
        from tsensei_v3_2_pianist.mjlab_env.preparation import options
        for extra in ({'lift':.02},{'hover':-1},{'settle_time':float('nan')}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                options(dict(mode='first_note_v1',duration=2,lift=0,**{k:v for k,v in extra.items() if k!='lift'}) if 'lift' not in extra else dict(mode='first_note_v1',duration=2,**extra))
