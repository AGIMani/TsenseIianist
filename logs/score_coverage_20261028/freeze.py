"""Freeze only the current task's reviewed overlays on a verified parent."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

root, parent, output = map(Path, sys.argv[1:4])
expected = '5a5a21db04d3f969b2052956568bb02d47368b457eb5760ab18dfd29971ceaa1'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

assert digest(parent) == expected
spec = importlib.util.spec_from_file_location('pack', root/'scripts/hpc/package_a32.py')
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
files = {}
with tarfile.open(parent) as archive:
    for member in archive:
        if member.isfile() and member.name != 'BUNDLE_MANIFEST.json':
            files[member.name] = archive.extractfile(member).read()
changes = {}
for name, sha in json.loads((root/'logs/score_coverage_20261028/overlay_manifest.json').read_text()).items():
    data = (root/name).read_bytes()
    assert pack.sha256(data) == sha, name
    old = files.get(name)
    if old != data:
        changes[name] = dict(parent_sha256=pack.sha256(old) if old else None, sha256=sha)
    files[name] = data
lineage = dict(parent_sha256=expected, overlays=changes,
    rule='Verified parent plus explicit score coverage validation/reporting overlays; model, original score, sampling and reward formula unchanged',
    formal_training_started=False)
files['docs/score_coverage_source_lineage_20261028.json'] = pack.canonical_json(lineage)
output.parent.mkdir(parents=True, exist_ok=True)
pack.write_archive(output, files, pack.build_manifest(files))
sha = digest(output)
output.with_suffix(output.suffix+'.sha256').write_text(sha+'  '+output.name+'\n')
output.with_suffix(output.suffix+'.lineage.json').write_bytes(pack.canonical_json(lineage))
audit = pack.audit_archive(output)
output.with_suffix(output.suffix+'.audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(dict(archive=str(output), sha256=sha, files=audit['file_count'], overlays=len(changes))), flush=True)
