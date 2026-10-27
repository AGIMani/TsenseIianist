"""Freeze only enumerated reviewed overlays on the verified current parent."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

root,parent,output=map(Path,sys.argv[1:4])
expected='63d4224c3a29807a6026d391967297ead55f163f221f8d29cc86f1b116bcb992'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
assert digest(parent)==expected
spec=importlib.util.spec_from_file_location('pack',root/'scripts/hpc/package_a32.py')
pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
files={}
with tarfile.open(parent) as archive:
    for member in archive:
        if member.isfile() and member.name!='BUNDLE_MANIFEST.json':
            files[member.name]=archive.extractfile(member).read()
changes={}
for name,sha in json.loads((root/'overlay_manifest.json').read_text()).items():
    data=(root/name).read_bytes();assert pack.sha256(data)==sha,name
    old=files.get(name)
    if old!=data:changes[name]=dict(parent_sha256=pack.sha256(old) if old else None,sha256=sha)
    files[name]=data
lineage=dict(parent_sha256=expected,overlays=changes,training_started=False,
    rule='Pinned parent plus enumerated key-hold source/tests/docs and current default reward JSON; no model/driver/score/budget changes')
files['docs/key_hold_lateral_source_lineage_20261026.json']=pack.canonical_json(lineage)
output.parent.mkdir(parents=True,exist_ok=True)
pack.write_archive(output,files,pack.build_manifest(files))
sha=digest(output)
output.with_suffix(output.suffix+'.sha256').write_text(sha+'  '+output.name+'\n')
(output.parent/'source_lineage.json').write_bytes(pack.canonical_json(lineage))
audit=pack.audit_archive(output)
(output.parent/'source_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(archive=str(output),sha256=sha,files=audit['file_count'],overlays=len(changes))),flush=True)
