import hashlib,importlib.util,json,tarfile
from pathlib import Path
root=Path.cwd();parent=root/'logs/black_key_model_20261023/source_black_key_shape_v1.tar.gz';expected='1951eb00657ad1a82a8bc7092f6274dcf4607f26746cc4dfa4d936f89761dd84'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(parent.read_bytes())==expected
with tarfile.open(parent) as a:files={m.name:a.extractfile(m).read() for m in a if m.isfile() and m.name!='BUNDLE_MANIFEST.json'}
changes={}
for name in ('tsensei_v3_2_pianist/mjlab_env/finger_coupling.py','tests/test_mesh_key_coupling_plane.py'):
 changes[name]=dict(before=sha(files[name]) if name in files else None,after=sha((root/name).read_bytes()));files[name]=(root/name).read_bytes()
spec=importlib.util.spec_from_file_location('pack',root/'scripts/hpc/package_a32.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=root/'logs/mesh_coupling_plane_20261023/source_mesh_coupling_plane_v1.tar.gz';out.parent.mkdir(exist_ok=True);assert not out.exists();manifest=m.build_manifest(files);m.write_archive(out,files,manifest);m.audit_archive(out);digest=sha(out.read_bytes());Path(str(out)+'.sha256').write_text(digest+'  '+out.name+'\n');(out.parent/'source_provenance.json').write_text(json.dumps(dict(parent_sha256=expected,sha256=digest,changes=changes),indent=2)+'\n');print(digest,flush=True)
