"""Remote-only: freeze the verified parent plus the exact reviewed overlays."""
import hashlib,importlib.util,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
parent=Path('/home/houj/projects/pianist_exper034_20261022/logs/replay_labels_removed_20261023/source_replay_key_colors_v1.tar.gz')
expected='cc965e669d495bc59f5166411927c381c5330542e41890f342e8cf64170d0849'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(parent.read_bytes())==expected
for name in ('cpu_tests','reference_tests','color_selected','runtime_target','export311'):
 assert (root/'validation'/f'{name}.exit').read_text().strip()=='0',name
with tarfile.open(parent) as archive:
 files={member.name:archive.extractfile(member).read() for member in archive if member.isfile() and member.name!='BUNDLE_MANIFEST.json'}
original={name:sha(data) for name,data in files.items()}
for name in (root/'overlay_files.txt').read_text().splitlines():
 files[name]=(root/'overlay'/name).read_bytes()
files['docs/change_log.md']+=(root/'change_log_entry.txt').read_bytes()
package=root/'package_pinned.py';package.write_bytes(files['scripts/hpc/package_a32.py'])
spec=importlib.util.spec_from_file_location('package_pinned',package);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
files=dict(sorted(files.items()));manifest=module.build_manifest(files)
changes={name:dict(before=original.get(name),after=sha(data)) for name,data in files.items() if original.get(name)!=sha(data)}
out=root/'source_black_key_shape_v1.tar.gz';assert not out.exists()
module.write_archive(out,files,manifest);del files
module.audit_archive(out)
checksum=sha(out.read_bytes())
Path(str(out)+'.sha256').write_text(checksum+'  '+out.name+'\n')
Path(str(out)+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
record=dict(parent=str(parent),parent_sha256=expected,archive=out.name,sha256=checksum,changed_members=changes,
 no_training=True,validation='validation/',remote_root=str(root),geometry='upper10x90_bottom12x95_height18_rear_aligned',
 unchanged='all key body/hinge positions, dynamics parameters and fingering reference targets')
(root/'source_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
metadata_path=root/'static_preview/preview.json';metadata=json.loads(metadata_path.read_text());metadata['source_archive_sha256']=checksum;metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(dict(archive=out.name,sha256=checksum,changed_members=len(changes))),flush=True)
