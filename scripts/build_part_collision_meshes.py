#!/usr/bin/env python3
"""Build body-local convex meshes per source CAD part, pruning only containment.

Remote CPU utility. No scaling down, trimming, or collision-mask changes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vertices(path):
    raw = path.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    dtype = np.dtype([('normal','<f4',(3,)),('points','<f4',(3,3)),('attr','<u2')])
    if len(raw) != 84 + count * 50:
        raise ValueError('Expected binary STL: '+str(path))
    return np.frombuffer(raw, dtype=dtype, offset=84)['points'].reshape(-1,3).astype(float)


def write_hull(path, points, hull):
    triangles = points[hull.simplices].copy()
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    for i in np.flatnonzero(np.sum(normals*hull.equations[:,:3],axis=1)<0):
        triangles[i,[1,2]] = triangles[i,[2,1]]
    with path.open('wb') as f:
        f.write(b'CAD part convex hull; no trimming'.ljust(80,b' '))
        f.write(struct.pack('<I',len(triangles)))
        for tri in triangles:
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
            f.write(struct.pack('<12fH',*normal,*tri.flatten(),0))


def outer_approximation(points, tolerance, padding):
    """Keep supporting planes: simplification can expand, never shrink a part."""
    hull=ConvexHull(points)
    equations=hull.equations.copy();equations[:,3]-=padding
    center=points[hull.vertices].mean(0)
    axes=np.concatenate((np.eye(3),-np.eye(3)))
    bounds=np.c_[axes,-np.max(points@axes.T,axis=0)-padding]
    chosen=set()
    for _ in range(len(equations)+1):
        planes=np.vstack((bounds,equations[sorted(chosen)])) if chosen else bounds
        v=HalfspaceIntersection(planes,center).intersections
        errors=v@equations[:,:3].T+equations[:,3]
        worst=np.argmax(errors,axis=1)
        extra=set(worst[np.max(errors,axis=1)>tolerance])-chosen
        if not extra:
            return v,ConvexHull(v),float(np.max(errors))
        chosen.update(extra)
    raise RuntimeError('Supporting-plane simplification did not converge')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--links',nargs='+',default=['base_link'])
    p.add_argument('--decompose-base-part',type=int,default=-1)
    p.add_argument('--concavity',type=float,default=.05)
    p.add_argument('--max-part-hulls',type=int,default=24)
    p.add_argument('--simplify-tolerance',type=float,default=0.)
    p.add_argument('--decomposition-padding',type=float,default=0.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    root=ET.parse(a.scene).getroot()
    for e in root.findall('asset/*'):
        if e.get('file'):
            e.set('file',str((a.scene.parent/e.get('file')).resolve()))
    report={}
    for side in ('left','right'):
        urdf=a.source/'urdf'/f'TsenseI_hand_{side}_v3_2_withid.urdf'
        ur=ET.parse(urdf).getroot()
        for label in a.links:
            name=label.replace('{side}',side)
            link=ur.find(f"link[@name='{name}']")
            candidates=[]
            for i,geo in enumerate(link.findall('collision')):
                mesh=geo.find('geometry/mesh');path=(urdf.parent/mesh.get('filename')).resolve()
                v=vertices(path)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                origin=geo.find('origin')
                if origin is not None:
                    rot=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
                    v=rot.apply(v)+np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                v, inverse=np.unique(v.round(12),axis=0,return_inverse=True)
                if name == 'base_link' and i == a.decompose_base_part:
                    import coacd
                    import importlib.metadata
                    coacd.set_log_level('warn')
                    cache=a.output/f'{side}_{name}_source_{i:03}_decomposition.npz'
                    cache_meta=cache.with_suffix('.json')
                    cache_identity=dict(source_sha256=sha(path),vertices_sha256=hashlib.sha256(v.tobytes()).hexdigest(),
                        concavity=a.concavity,max_part_hulls=a.max_part_hulls,
                        coacd=importlib.metadata.version('coacd'),seed=0,preprocess_mode='off',
                        resolution=2000,mcts_nodes=20,mcts_iterations=100,mcts_max_depth=3)
                    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())==cache_identity:
                        with np.load(cache) as saved:
                            chunks=[saved[k] for k in sorted(saved.files)]
                    else:
                        result=coacd.run_coacd(coacd.Mesh(v,inverse.reshape(-1,3)),
                            threshold=a.concavity,max_convex_hull=a.max_part_hulls,
                            preprocess_mode='off',resolution=2000,
                            mcts_nodes=20,mcts_iterations=100,mcts_max_depth=3,
                            seed=0,merge=True)
                        chunks=[np.asarray(points) for points,faces in result]
                        np.savez_compressed(cache,**{f'v{j:03}':vs for j,vs in enumerate(chunks)})
                        cache_meta.write_text(json.dumps(cache_identity,indent=2)+'\n')
                    print(f'{side}/{name} source {i}: {len(chunks)} convex pieces',flush=True)
                else:chunks=[v]
                for j,points in enumerate(chunks):
                    h=ConvexHull(points)
                    candidates.append(dict(i=i if len(chunks)==1 else i*1000+j,
                        source_index=i,v=points,h=h,path=str(path.relative_to(a.source)),sha256=sha(path)))
            candidates.sort(key=lambda c:-c['h'].volume)
            retained=[];covered={}
            for c in candidates:
                ex=c['v'][c['h'].vertices]
                container=next((b for b in retained if
                    np.all(ex.min(0)>=b['v'].min(0)-1e-7) and
                    np.all(ex.max(0)<=b['v'].max(0)+1e-7) and
                    np.max(ex@b['h'].equations[:,:3].T+b['h'].equations[:,3])<=1e-7),None)
                if container is None:retained.append(c)
                else:covered[c['i']]=container['i']
            body=root.find(f".//body[@name='{side}_/{name}']")
            original=body.find(f"geom[@name='{side}_/{name}_collision_0']")
            parts=[]
            for c in retained:
                approximation_error=0.
                pad=a.decomposition_padding if c['source_index']==a.decompose_base_part else 0.
                if a.simplify_tolerance or pad:
                    c['v'],c['h'],approximation_error=outer_approximation(c['v'],a.simplify_tolerance,pad)
                meshname=f'{side}_{name}_part_{c["i"]:03}'
                path=a.output/(meshname+'.stl');write_hull(path,c['v'],c['h'])
                ET.SubElement(root.find('asset'),'mesh',name=meshname,file=str(path.resolve()))
                geom=copy.deepcopy(original);geom.set('mesh',meshname)
                geom.set('name',f'{side}_/{name}_collision_{len(parts)}');body.append(geom)
                parts.append(dict(source_index=c['source_index'],piece_index=c['i'],mesh_file=path.name,hull_sha256=sha(path),
                    source_file=c['path'],source_sha256=c['sha256'],volume_m3=c['h'].volume,
                    vertices=len(c['h'].vertices),triangles=len(c['h'].simplices),
                    outward_padding_m=pad,max_support_plane_excess_m=approximation_error))
            body.remove(original)
            report[side+'/'+name]=dict(parts=parts,source_parts=len(link.findall('collision')),
                candidate_hulls=len(candidates),contained_parts=covered,
                decomposition=dict(source_index=a.decompose_base_part,concavity=a.concavity,
                                   max_part_hulls=a.max_part_hulls,preprocess_mode='off',seed=0),
                source_urdf_sha256=sha(urdf))
    ET.ElementTree(root).write(a.output/'scene.xml')
    (a.output/'parts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:dict(source_parts=v['source_parts'],retained=len(v['parts'])) for k,v in report.items()}))


if __name__=='__main__':main()
