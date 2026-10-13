"""CPU Warp tests for contact occurrence, body-pair deduplication and reset isolation."""
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch
import warp as wp

from tsensei_v3_2_pianist.mjlab_env.self_collision import SelfCollisionTracker


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="right_/palm"><joint type="slide"/><geom name="rp" size=".04"/>
        <body name="right_/fixed"><geom name="rf" size=".02"/></body></body>
      <body name="right_/index"><joint type="slide"/><geom name="ri" size=".02"/>
        <geom name="ri2" size=".015"/></body>
      <body name="left_/palm"><joint type="slide"/><geom name="lp" size=".04"/></body>
      <body name="piano/key"><joint type="slide"/><geom name="pk" size=".02"/></body>
    </worldbody></mujoco>''')


class CollisionDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.model=model()
        self.tracker=SelfCollisionTracker(self.model,2,'cpu',categories=('self','inter_hand'),diagnostics=True)

    def sample(self, contacts, *, count_substep=True):
        # Synthetic contact buffers test counting only; real contact generation
        # and model exclusions remain MuJoCo's responsibility.
        geom=np.full((8,2),-1,dtype=np.int32)
        distance=np.ones(8,dtype=np.float32)
        world=np.zeros(8,dtype=np.int32)
        for i,(a,b,w,dist) in enumerate(contacts):
            geom[i]=[self.model.geom(a).id,self.model.geom(b).id]
            distance[i]=dist;world[i]=w
        data=SimpleNamespace(naconmax=8,
            nacon=wp.array([len(contacts)],dtype=wp.int32,device='cpu'),
            contact=SimpleNamespace(geom=wp.array(geom,dtype=wp.vec2i,device='cpu'),
                dist=wp.array(distance,dtype=wp.float32,device='cpu'),
                worldid=wp.array(world,dtype=wp.int32,device='cpu')))
        sim=SimpleNamespace(wp_data=data,wp_device='cpu')
        self.tracker.sample(sim,count_substep=count_substep)
        wp.synchronize_device('cpu')

    def test_duplicate_contacts_keys_welds_and_positive_distance(self):
        self.sample([('rp','ri',0,-.001),('rp','ri2',0,-.002),
                     ('rp','lp',0,0.),('ri','pk',1,-.1),
                     ('rp','rf',1,-.1),('ri','lp',1,.001)])
        self.assertEqual(self.tracker.count(('self',)).tolist(),[1,0])
        self.assertEqual(self.tracker.count(('inter_hand',)).tolist(),[1,0])
        self.assertEqual(self.tracker.substep_counts.tolist(),[[1,1,1,1],[0,0,0,1]])
        self.sample([('rp','ri',0,-.01),('ri','lp',1,-.001)])
        self.assertEqual(self.tracker.count(('self',)).tolist(),[1,0])
        self.assertEqual(self.tracker.count(('inter_hand',)).tolist(),[1,1])
        self.assertEqual(self.tracker.substep_counts.tolist(),[[2,1,2,2],[0,1,1,2]])
        self.sample([])
        self.assertEqual(self.tracker.substep_counts.tolist(),[[2,1,2,3],[0,1,1,3]])

    def test_final_forward_union_does_not_add_a_physics_step_and_partial_reset(self):
        self.sample([('rp','ri',0,-.001),('rp','lp',1,-.001)])
        before=self.tracker.substep_counts.clone()
        self.sample([('rp','ri',1,-.001)],count_substep=False)
        torch.testing.assert_close(self.tracker.substep_counts,before)
        self.assertEqual(self.tracker.count(('self',)).tolist(),[1,1])
        other_seen=self.tracker.seen[1].clone()
        self.tracker.clear(torch.tensor([0]))
        self.assertEqual(self.tracker.seen[0].count_nonzero().item(),0)
        self.assertEqual(self.tracker.substep_counts[0].tolist(),[0,0,0,0])
        torch.testing.assert_close(self.tracker.seen[1],other_seen)
        torch.testing.assert_close(self.tracker.substep_counts[1],before[1])
        self.tracker.clear()
        self.assertEqual(self.tracker.substep_counts.count_nonzero().item(),0)

    def test_default_tracker_allocates_no_diagnostic_arrays(self):
        tracker=SelfCollisionTracker(self.model,2,'cpu')
        self.assertFalse(tracker.diagnostics)
        self.assertFalse(hasattr(tracker,'substep_counts'))
        self.assertFalse(hasattr(tracker,'substep_present'))
        self.assertTrue(all(c=='self' for c in tracker.pair_categories))


if __name__=='__main__':
    unittest.main()
