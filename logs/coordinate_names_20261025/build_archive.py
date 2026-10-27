"""Freeze only this task's reviewed coordinate-name overlay on the pinned parent."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

root, parent, output = map(Path, sys.argv[1:4])
expected_parent = 'f9741e1f68b6253909011edde71993bc252a6829f61fda4840c095154467bccf'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == expected_parent
spec = importlib.util.spec_from_file_location('pack', root / 'scripts/hpc/package_a32.py')
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
files = {}
with tarfile.open(parent) as archive:
    for member in archive:
        if member.isfile() and member.name != 'BUNDLE_MANIFEST.json':
            files[member.name] = archive.extractfile(member).read()
changes = {}
for name, expected in json.loads((root / 'overlay_manifest.json').read_text()).items():
    data = (root / name).read_bytes()
    assert pack.sha256(data) == expected, name
    old = files.get(name)
    if old != data:
        changes[name] = dict(parent_sha256=pack.sha256(old) if old else None, sha256=expected)
    files[name] = data
lineage = dict(parent_sha256=expected_parent, overlays=changes,
    rule='Only enumerated coordinate naming/world initial pose files overlaid; all parent URDF/mesh/reward-JSON bytes retained',
    training_started=False)
files['docs/coordinate_names_source_lineage_20261025.json'] = pack.canonical_json(lineage)
output.parent.mkdir(parents=True, exist_ok=True)
pack.write_archive(output, files, pack.build_manifest(files))
sha = pack.sha256(output.read_bytes())
output.with_suffix(output.suffix + '.sha256').write_text(sha + '  ' + output.name + '\n')
(output.parent / 'source_lineage.json').write_bytes(pack.canonical_json(lineage))
audit = pack.audit_archive(output)
(output.parent / 'source_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(dict(archive=str(output), sha256=sha, files=audit['file_count'], overlays=len(changes))), flush=True)
