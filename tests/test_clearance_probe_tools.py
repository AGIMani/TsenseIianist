"""CPU contracts for explicit migration probes and step-diagnostic recording."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


probe=module('clearance_probe','probe_clearance_contacts.py')
audit=module('clearance_audit','audit_checkpoint_trajectories.py')


class ClearanceProbeToolsTest(unittest.TestCase):
    def test_baseline_preserves_old_options_and_candidate_drops_old_x_ranges(self):
        parent={'control_limits':{'channels':{'L_Wrist_TX':{'min':-.1,'max':.2}}},
                'reward_config':{'terms':{'old':True}},'preparation':{'mode':'policy_v1'}}
        baseline=probe.variant_options(parent)
        self.assertEqual(baseline['control_limits'],parent['control_limits'])
        self.assertEqual(baseline['reward_config'],parent['reward_config'])
        candidate=probe.variant_options(parent,candidate=True,clearance={'schema':1},reward={'terms':{'new':True}})
        self.assertIsNone(candidate['control_limits'])
        self.assertEqual(candidate['finger_clearance'],{'schema':1})
        self.assertEqual(candidate['reward_config'],{'terms':{'new':True}})
        self.assertEqual(parent['control_limits']['channels']['L_Wrist_TX']['min'],-.1)
        self.assertTrue(baseline['collision_diagnostics'])
        args=probe.parser().parse_args(['--checkpoint','best.pt','--output','out'])
        self.assertEqual(args.repeats,3)
        self.assertEqual(args.detail_repeats,1)

    def test_physical_hashes_cannot_be_changed_by_probe(self):
        saved={'scene_sha256':'scene','task_sha256':'task'}
        probe.check_physical_contract('scene','task',saved)
        for scene,task in (('bad','task'),('scene','bad')):
            with self.subTest(scene=scene,task=task),self.assertRaisesRegex(ValueError,'identical parent'):
                probe.check_physical_contract(scene,task,saved)

    def test_step_wrapper_forwards_assigned_keys_and_restores_callback_contract(self):
        assigned=object();callback=object();seen={}
        def step(action,**kwargs):seen.update(action=action,**kwargs);return 23
        wrapped=audit.attach_substep_capture(step,callback)
        self.assertEqual(wrapped('action',assigned_keys=assigned),23)
        self.assertIs(seen['assigned_keys'],assigned)
        self.assertIs(seen['substep_callback'],callback)
        with self.assertRaisesRegex(ValueError,'nest'):wrapped('action',substep_callback=object())

    def test_diagnostic_summary_keeps_actual_and_request_distinct(self):
        values={'finger_actual_deficit_m':np.array([[0.,.002],[0.,0.]]),
                'finger_requested_deficit_m':np.array([[.01,.001],[.01,0.]]),
                'finger_projection_mask':np.array([[True,False],[True,True]]),
                'requested_physical_action':np.array([[.1,.3],[.2,.4]]),
                'applied_physical_action':np.array([[.05,.3],[.1,.2]])}
        result=audit.diagnostic_summary(values,frames=2,physical_action_names=['R_Wrist_TX','R_Thumb_PIP'])
        self.assertEqual(result['fields']['finger_actual_deficit_m']['above_1um_frames'],1)
        self.assertEqual(result['fields']['finger_requested_deficit_m']['above_1um_frames'],2)
        self.assertEqual(result['fields']['finger_projection_mask']['channel_frame_fraction'],.75)
        self.assertEqual(result['requested_applied_difference']['per_channel'][0]['unit'],'m')
        self.assertEqual(result['requested_applied_difference']['per_channel'][1]['unit'],'rad')
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            audit.diagnostic_summary(values,frames=3,physical_action_names=[])


if __name__=='__main__':unittest.main()
