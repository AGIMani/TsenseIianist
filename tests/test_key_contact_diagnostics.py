"""CPU Warp contact classification, orientation and local-reset checks."""
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch
import warp as wp

from tsensei_v3_2_pianist.mjlab_env.key_contact_diagnostics import KeyContactDiagnostics


class KeyContactChecks(unittest.TestCase):
    def setUp(self):
        self.model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="piano/white_key_geom_0" type="box" size=".1 .02 .01"/>
          <geom name="piano/black_key_geom_1" type="box" size=".07 .01 .02"/>
          <body><joint type="slide"/><geom name="right_/right_hand_index_DIP_link_collision_0" type="capsule" size=".004 .01"/></body>
          <body><joint type="slide"/><geom name="left_/left_hand_thumb_PIP_link_collision_0" type="capsule" size=".004 .01"/></body>
          <body><joint type="slide"/><geom name="left_/left_hand_index_MCP_link_collision_0" size=".008"/></body>
          <body><joint type="slide"/><geom name="right_/Right_hand_palm_link_collision_0" size=".02"/></body>
          <geom name="piano/case" size=".1"/>
        </worldbody></mujoco>''')
        self.tracker=KeyContactDiagnostics(self.model,2,'cpu')
        self.ids=[self.model.geom(n).id for n in ('piano/white_key_geom_0','piano/black_key_geom_1',
            'right_/right_hand_index_DIP_link_collision_0','left_/left_hand_thumb_PIP_link_collision_0',
            'left_/left_hand_index_MCP_link_collision_0','right_/Right_hand_palm_link_collision_0','piano/case')]

    def sample(self,rows,rotations=None,*,count_substep=True):
        count=16;geom=np.full((count,2),-1,dtype=np.int32);world=np.zeros(count,dtype=np.int32)
        distance=np.ones(count,dtype=np.float32);frame=np.tile(np.eye(3,dtype=np.float32),(count,1,1))
        for i,(a,b,w,dist,normal) in enumerate(rows):
            geom[i]=[self.ids[a] if a>=0 else a,self.ids[b] if b>=0 else b]
            world[i]=w;distance[i]=dist;frame[i,0]=normal
        matrices=np.tile(np.eye(3,dtype=np.float32),(2,self.model.ngeom,1,1))
        for w,g,matrix in rotations or []:matrices[w,self.ids[g]]=matrix
        data=SimpleNamespace(naconmax=count,nacon=wp.array([len(rows)],dtype=wp.int32,device='cpu'),
            geom_xmat=wp.array(matrices,dtype=wp.mat33,device='cpu'),
            contact=SimpleNamespace(geom=wp.array(geom,dtype=wp.vec2i,device='cpu'),
                dist=wp.array(distance,dtype=wp.float32,device='cpu'),
                worldid=wp.array(world,dtype=wp.int32,device='cpu'),
                frame=wp.array(frame,dtype=wp.mat33,device='cpu')))
        before=[a.numpy().copy() for a in (data.contact.geom,data.contact.dist,data.contact.frame,data.geom_xmat)]
        self.tracker.sample(SimpleNamespace(wp_data=data,wp_device='cpu'),count_substep=count_substep)
        wp.synchronize_device('cpu')
        for a,b in zip(before,(data.contact.geom,data.contact.dist,data.contact.frame,data.geom_xmat)):
            np.testing.assert_array_equal(a,b.numpy())

    def test_top_side_underside_dedup_and_non_distal(self):
        self.sample([(0,2,0,-.001,[0,0,-1]),(0,3,0,0,[0,0,-1]),
                     (1,4,0,-.002,[1,0,0]),(0,5,1,-.001,[0,0,1]),
                     (0,2,1,.001,[0,0,-1]),(2,3,1,-.01,[0,0,-1]),(6,2,1,-.01,[0,0,-1])])
        self.assertEqual(self.tracker.substep_counts.tolist(),[[1,1,1,1],[0,0,1,1]])
        self.sample([])
        self.assertEqual(self.tracker.substep_counts.tolist(),[[1,1,1,2],[0,0,1,2]])

    def test_normal_reversed_geometry_order_and_rotated_key(self):
        # Key local +Z is world +X. A world -X normal from key to hand is underside.
        ry=np.array([[0,0,1],[0,1,0],[-1,0,0]],dtype=np.float32)
        self.sample([(1,2,0,-.001,[-1,0,0]),(2,1,1,-.001,[1,0,0])],[(0,1,ry),(1,1,ry)])
        self.assertEqual(self.tracker.substep_counts.tolist(),[[1,1,0,1],[1,1,0,1]])

    def test_threshold_boundaries_invalid_contacts_and_partial_reset(self):
        self.sample([(0,2,0,-.001,[.7141428,0,-.7]),(0,3,1,-.001,[.7141428,0,.7]),
                     (-1,2,1,-.1,[0,0,-1]),(0,2,-1,-.1,[0,0,-1])])
        self.assertEqual(self.tracker.substep_counts.tolist(),[[1,1,0,1],[0,0,0,1]])
        before=self.tracker.substep_counts.clone()
        self.sample([(0,2,1,-.01,[0,0,-1])],count_substep=False)
        torch.testing.assert_close(before,self.tracker.substep_counts)
        self.tracker.clear(torch.tensor([0]))
        self.assertEqual(self.tracker.substep_counts[0].tolist(),[0,0,0,0])
        torch.testing.assert_close(self.tracker.substep_counts[1],before[1])
        self.tracker.clear();self.assertEqual(self.tracker.substep_counts.count_nonzero(),0)

    def test_explicit_mapping_validation_and_contract(self):
        self.assertEqual(len(self.tracker.contract['terminal_capsule_names']),2)
        self.assertEqual(len(self.tracker.contract['non_distal_geom_names']),2)
        self.assertEqual(self.tracker.contract['key_local_outward_normal']['underside_z_le'],-.7)
        with self.assertRaisesRegex(ValueError,'key geometry'):
            KeyContactDiagnostics(self.model,1,'cpu',key_geom_names=['missing'])

    def test_a32_housing_underside_is_counted_without_mount_or_visual_geometry(self):
        self.model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="piano/white_key_geom_0" type="box" size=".1 .02 .01"/>
          <geom name="right_/base_link_collision_0" type="box" size=".02 .02 .01"/>
          <geom name="left_/base_link_collision_0" type="box" size=".02 .02 .01"/>
          <geom name="right_/base_link_visual_0" size=".02" contype="0" conaffinity="0"/>
          <geom name="left_/wrist_mount_collision_0" size=".02"/>
        </worldbody></mujoco>''')
        self.tracker=KeyContactDiagnostics(self.model,2,'cpu')
        self.ids=list(range(self.model.ngeom))
        self.assertEqual(self.tracker.contract['non_distal_geom_names'],
                         ['right_/base_link_collision_0','left_/base_link_collision_0'])
        # Both orderings; visual/mount contacts in world 1 must be excluded.
        self.sample([(0,1,0,-.001,[0,0,-1]),(2,0,0,-.001,[0,0,1]),
                     (0,3,1,-.001,[0,0,-1]),(0,4,1,-.001,[0,0,-1])])
        self.assertEqual(self.tracker.substep_counts.tolist(),[[1,1,1,1],[0,0,0,1]])


if __name__=='__main__':unittest.main()
