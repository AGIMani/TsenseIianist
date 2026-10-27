"""Lightweight ending contract/timeline tests (no scene construction)."""
import unittest
import numpy as np
from tsensei_v3_2_pianist.mjlab_env import ending

class EndingTests(unittest.TestCase):
    def test_release_then_return_original_with_bounded_commands(self):
        start=np.array([.8,.4]); origin=np.array([.1,0.])
        t,q,release=ending.plan(start,origin,np.array([False,True]),np.array([3.,.3]),np.array([15.,.6]),2.,.05,.001,.5)
        np.testing.assert_array_equal(q[0],start)
        np.testing.assert_allclose(q[-1],origin,atol=1e-12)
        np.testing.assert_allclose(q[t<=release,1],start[1])
        self.assertTrue((abs(np.diff(q,axis=0)/.001).max(axis=0)<=[3.00001,.30001]).all())
        self.assertTrue((abs(np.diff(q,n=2,axis=0)/.001**2).max(axis=0)<=[15.001,.601]).all())
    def test_append_checks_seam_and_avoids_duplicate(self):
        raw={'qpos':np.array([[1.],[2.]]),'qvel':np.zeros((2,1)), 'sustain':np.ones(2)}
        trace={'qpos':np.array([[2.],[3.],[4.]]),'qvel':np.zeros((3,1)),'time':np.array([0.,.1,.2])}
        out=ending.append_trace(raw,trace,.1)
        np.testing.assert_array_equal(out['qpos'][:,0],[1,2,3,4])
        np.testing.assert_array_equal(out['sustain'],[1,1,0,0])
        trace['qpos'][0]=9
        with self.assertRaises(ValueError):ending.append_trace(raw,trace,.1)
    def test_contract_requires_first_note_and_finite_time(self):
        from tsensei_v3_2_pianist.mjlab_env.preparation import options
        base=dict(mode='first_note_v1',duration=2,lift=0,ending_duration=2)
        self.assertEqual(options(base)['ending_duration'],2)
        for bad in (dict(base,ending_duration=-1),dict(base,ending_duration=float('nan')),dict(base,mode='fixed_v1')):
            with self.assertRaises(ValueError):options(bad)

    def test_empty_annotations_cover_return_without_extending_notes(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from export_score_annotations import append_ending
        arrays=dict(time=np.array([.05,.1]),goal=np.ones((2,89)),fingers=np.ones((2,10,88)))
        out=append_ending(arrays,.5,.05)
        self.assertAlmostEqual(out['time'][-1],.6)
        self.assertEqual(out['goal'][2:].sum(),0)
        self.assertEqual(out['fingers'][2:].sum(),0)
        np.testing.assert_array_equal(out['goal'][:2],arrays['goal'])

    def test_entry_freezes_ending_without_changing_learning_config(self):
        from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config
        base=['--no-arms','--disable-wrist-tz','--disable-wrist-yaw','--action-profile','direct_joint_v1',
            '--finger-coupling','--finger-coupling-angle-ranges','configs/finger_coupling/angle_ranges_10_80.json',
            '--preparation-mode','first_note_v1','--preparation-time','2','--preparation-lift','0']
        before=effective_config(parser().parse_args(base))
        after=effective_config(parser().parse_args(base+['--ending-time','2']))
        self.assertEqual(after['preparation'].pop('ending_duration'),2.)
        self.assertEqual(before,after)
        with self.assertRaises(ValueError):effective_config(parser().parse_args(['--ending-time','2']))

    def test_failed_render_still_flushes_physics(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from capture_archive_render import run_and_flush
        saved=[]
        def fail():raise ValueError('arrival failure')
        with self.assertRaisesRegex(ValueError,'arrival failure'):
            run_and_flush(fail,lambda error:saved.append(str(error)))
        self.assertEqual(saved,['arrival failure'])
