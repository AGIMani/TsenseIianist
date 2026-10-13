#!/usr/bin/env python3
"""Augment an existing frozen scene/task with 93 mm M6 IK and reward pairs."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def export(scene_path,task_path,output):
    import mujoco
    import numpy as np
    from dm_control import mjcf
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle,sha256
    from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle
    from tsensei_v3_2_pianist.preview_arm_ik import ArmPreview
    from tsensei_v3_2_pianist.preview_arm_collision import ArmForearmCollision
    source=SceneBundle(scene_path); task=TaskBundle(task_path,source)
    if 'm6' in source.manifest:
        raise ValueError('Input scene already contains M6; choose an original hand-only scene')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    scene_out=output/'scene';task_out=output/'task'
    scene_out.mkdir(parents=True);task_out.mkdir()
    # Only generated standalone arm assets are merged; input frozen bundle is immutable.
    root=mjcf.RootElement(model='m6_assets');arms=ArmPreview(root)
    xml=ET.parse(source.directory/'scene.xml')
    extra=ET.fromstring(root.to_xml_string())
    for element in extra.iter():
        if element.get('name','').startswith('//unnamed'):
            element.set('name','tj_extra_'+element.get('name').replace('/','_'))
    for section in ('asset','worldbody'):
        for element in extra.find(section):xml.getroot().find(section).append(copy.deepcopy(element))
    for name in source.manifest['files']:
        if name not in ('scene.xml','cpu_reference.npz'):
            (scene_out/name).write_bytes((source.directory/name).read_bytes())
    for name,content in root.get_assets().items():(scene_out/name).write_bytes(content)
    xml.write(scene_out/'scene.xml')
    physics=mjcf.Physics.from_xml_path(str(scene_out/'scene.xml'))
    physics.data.qpos[:]=source.reset_qpos;physics.data.qvel[:]=source.reset_qvel
    physics.data.ctrl[:]=source.reset_ctrl;physics.forward()
    hands=[SimpleNamespace(root_body=SimpleNamespace(full_identifier=source.manifest['hands'][s]['body']))
           for s in ('right','left')]
    arms.bind(physics,hands)
    collision=ArmForearmCollision(physics,arms)
    m=physics.model.ptr
    contact=xml.getroot().find('contact')
    if contact is None:contact=ET.SubElement(xml.getroot(),'contact')
    for a,b in collision.pairs:
        if not (m.geom(a).name.startswith('tj_collision_') or m.geom(b).name.startswith('tj_collision_')):continue
        ET.SubElement(contact,'pair',geom1=m.geom(a).name,geom2=m.geom(b).name,
                      margin='0',gap='0',condim='1')
    xml.write(scene_out/'collision_query.xml')
    for pair in list(contact):
        if pair.tag=='pair' and (pair.get('geom1','').startswith('tj_collision_') or
                                pair.get('geom2','').startswith('tj_collision_')):
            contact.remove(pair)
    xml.write(scene_out/'scene.xml')
    data=copy.deepcopy(source.manifest)
    data['stage']='music scene with GPU kinematic M6 followers and detection-only arm contacts'
    data['mocap_pos']=physics.data.mocap_pos.tolist();data['mocap_quat']=physics.data.mocap_quat.tolist()
    # Serialize FK constants, so training never imports Composer or the source URDF.
    fk=arms.model
    armdata={}
    zero=mujoco.MjData(m);zero.qpos[:]=physics.data.qpos
    for side,aids in arms.wrist_actuators.items():
        jids=m.actuator_trnid[aids,0];zero.qpos[m.jnt_qposadr[jids]]=0
    mujoco.mj_forward(m,zero)
    for side,arm in arms.sides.items():
        bids=[fk.jnt_bodyid[fk.joint(name).id] for name in arm['joints']]
        base=int(fk.body_parentid[bids[0]])
        def matrix(q):
            out=np.zeros(9);mujoco.mju_quat2Mat(out,q);return out.reshape(3,3).tolist()
        aids=arms.wrist_actuators[side];jids=m.actuator_trnid[aids,0];bid=arms.hand_ids[side]
        armdata[side]=dict(base_pos=fk.body_pos[base].tolist(),base_rot=matrix(fk.body_quat[base]),
            joint_pos=fk.body_pos[bids].tolist(),joint_rot=[matrix(fk.body_quat[b]) for b in bids],
            axis=fk.jnt_axis[[fk.joint(n).id for n in arm['joints']]].tolist(),
            limits=arm['limits'].tolist(),velocity=arm['velocity'].tolist(),q=arms.data.qpos[arm['qadr']].tolist(),
            tool_pos=fk.site_pos[arm['sid']].tolist(),tool_rot=matrix(fk.site_quat[arm['sid']]),
            mocap_ids=[int(m.body_mocapid[m.body(f'tj_{fk.body(b).name}').id]) for b in [base]+bids],
            hand_body=bid,hand_qadr=m.jnt_qposadr[jids].tolist(),hand_dof=m.jnt_dofadr[jids].tolist(),
            hand_axis=m.jnt_axis[jids].tolist(),hand_kind=m.jnt_type[jids].tolist(),
            hand_rest_pos=zero.xpos[bid].tolist(),hand_rest_rot=zero.xmat[bid].reshape(3,3).tolist(),
            handle_id=arms.handle_ids[side])
    data['m6']=dict(format='m6_bundle_current',config=arms.config,arms=armdata,collision=collision.manifest(),
        mode='GPU kinematic IK; detection-only contacts; original 51 actions',
        source_scene_sha256=sha256(source.directory/'manifest.json'))
    data['limitations'] += ['TJ-ARM-02/03 provisional mount', 'M6 contacts detect reward only; no arm forces']
    data['files']={p.name:sha256(p) for p in scene_out.iterdir() if p.is_file()}
    (scene_out/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    newtask=copy.deepcopy(task.manifest)
    for name in newtask['files']:
        if name!='cpu_reference.npz':(task_out/name).write_bytes((task.directory/name).read_bytes())
    newtask['scene_manifest_sha256']=sha256(scene_out/'manifest.json')
    newtask['forearm_collision_scope']='cross_hands_cross_arms_and_arm_hand_v1'
    newtask['forearm_collision_pairs']=data['m6']['collision']['pairs']
    newtask['cpu_reference_cases']=[]
    newtask['limitations'] += ['M6 extension has separate CPU/GPU checks; old CPU rollout not reused']
    newtask['files']={p.name:sha256(p) for p in task_out.iterdir() if p.is_file()}
    (task_out/'task.json').write_text(json.dumps(newtask,indent=2)+'\n')
    loaded=SceneBundle(scene_out);TaskBundle(task_out,loaded)
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True,type=Path);p.add_argument('--task',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();export(a.scene,a.task,a.output)
