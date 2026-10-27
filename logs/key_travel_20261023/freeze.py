import hashlib,importlib.util,json,tarfile
from pathlib import Path
root=Path.cwd();runtime=root/'runtime'
for name in ['model','tests']:assert (root/'validation'/f'{name}.exit').read_text().strip()=='0'
parent=json.loads((root/'parent.json').read_text());sha=lambda data:hashlib.sha256(data).hexdigest()
assert sha(Path(parent['path']).read_bytes())==parent['sha256']
with tarfile.open(parent['path']) as a:files={m.name:a.extractfile(m).read() for m in a if m.isfile() and m.name!='BUNDLE_MANIFEST.json'}
changes={}
for name in ['tsensei_v3_2_pianist/models/piano/piano_constants.py','tests/test_key_activation_fraction.py','README.md','AGENTS.md','docs/reward_defaults_20261006.md','docs/black_key_shape_20261023.md']:
 data=(runtime/name).read_bytes();changes[name]={'before':sha(files[name]),'after':sha(data)};files[name]=data
files['docs/change_log.md']+=(root/'change_log_entry.txt').read_bytes()
spec=importlib.util.spec_from_file_location('pack',runtime/'scripts/hpc/package_a32.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=root/'source_key_travel_12_10_v1.tar.gz';assert not out.exists();manifest=m.build_manifest(files);m.write_archive(out,files,manifest);m.audit_archive(out)
digest=sha(out.read_bytes());Path(str(out)+'.sha256').write_text(digest+'  '+out.name+'\n');Path(str(out)+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(root/'source_provenance.json').write_text(json.dumps({'parent':parent,'sha256':digest,'changes':changes,'training_started':False},indent=2)+'\n');print(digest,flush=True)
