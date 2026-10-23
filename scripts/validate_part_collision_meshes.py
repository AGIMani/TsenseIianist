#!/usr/bin/env python3
"""Remote CPU validation of repaired meshes against supplied CAD and dynamics."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from build_part_collision_meshes import vertices, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args()
    records=json.loads((a.candidate/'parts.json').read_text())
    coverage={}
    for key, record in records.items():
        side, name=key.split('/')
        urdf=a.source/'urdf'/f'TsenseI_hand_{side}_v3_2_withid.urdf'
        link=ET.parse(urdf).getroot().find(f"link[@name='{name}']")
        planes=[];triangles=0;verts=0
        for part in record['parts']:
            path=a.candidate/part['mesh_file']
            assert sha(path)==part['hull_sha256']
            v=vertices(path);h=ConvexHull(v);planes.append(h.equations)
            triangles+=len(v)//3;verts+=len(h.vertices)
        max_error=0.;count=0;outside=0
        for geo in link.findall('collision'):
            mesh=geo.find('geometry/mesh');path=(urdf.parent/mesh.get('filename')).resolve()
            v=vertices(path)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            # CAD vertices and every triangle centroid, not just bounding boxes.
            v=np.concatenate((v,np.mean(v.reshape(-1,3,3),axis=1)))
            o=geo.find('origin')
            if o is not None:
                v=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).apply(v)
                v+=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
            v=np.unique(v.round(12),axis=0)
            for batch in np.array_split(v,max(1,len(v)//512)):
                errors=np.full(len(batch),np.inf)
                for eq in planes:
                    errors=np.minimum(errors,np.max(batch@eq[:,:3].T+eq[:,3],axis=1))
                max_error=max(max_error,float(errors.max()));count+=len(errors)
                outside+=int(np.sum(errors>1e-6))
        coverage[key]=dict(samples=count,outside_1um=outside,
            max_halfspace_violation_m=max_error,convex_pieces=len(planes),triangles=triangles,vertices=verts)
    baseline=mujoco.MjModel.from_binary_path(str(a.baseline))
    candidate=mujoco.MjModel.from_xml_path(str(a.candidate/'scene.xml'))
    invariants={}
    # XML reserialization rounds stored values: compare baseline XML separately
    # to isolate collider-only changes, with byte-identical dynamics arrays.
    baseline_xml=mujoco.MjModel.from_xml_path(str(a.baseline.with_name('scene.xml')))
    for field in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
                  'jnt_axis','jnt_pos','jnt_range','dof_damping','dof_armature',
                  'actuator_ctrlrange','actuator_forcerange','actuator_gainprm','exclude_signature'):
        invariants[field]=bool(np.array_equal(getattr(baseline_xml,field),getattr(candidate,field)))
    mujoco.mj_saveModel(candidate,str(a.candidate/'model.mjb'),None)
    report=dict(coverage=coverage,invariants=invariants,
        baseline_ngeom=baseline.ngeom,candidate_ngeom=candidate.ngeom,
        baseline_nq=baseline.nq,candidate_nq=candidate.nq,
        candidate_scene_sha256=sha(a.candidate/'scene.xml'))
    report['passed']=all(v['outside_1um']==0 for v in coverage.values()) and all(invariants.values())
    (a.candidate/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
