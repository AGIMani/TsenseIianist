"""Freeze parent archive plus the explicit, reviewed ending return overlay."""
from pathlib import Path
import sys,tarfile,json,hashlib,importlib.util
root=Path(sys.argv[1]);parent=Path(sys.argv[2]);output=Path(sys.argv[3])
spec=importlib.util.spec_from_file_location('pack',root/'scripts/hpc/package_a32.py');pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
files={}
with tarfile.open(parent) as archive:
 for item in archive:
  if item.isfile() and item.name!='BUNDLE_MANIFEST.json':files[item.name]=archive.extractfile(item).read()
overlays=['AGENTS.md', 'README.md', 'docs/change_log.md', 'docs/reward_defaults_20261006.md', 'docs/superpowers/plans/2026-10-25-ending-return.md', 'docs/superpowers/specs/2026-10-25-ending-return.md', 'preview/scripts/render_mjlab_checkpoint.py', 'scripts/capture_archive_render.py', 'scripts/export_score_annotations.py', 'scripts/export_videos.py', 'scripts/package_interactive_replay.py', 'scripts/plot_recorded_midi_comparison.py', 'scripts/train_from_source.py', 'tests/test_ending_export_gpu.py', 'tests/test_ending_gpu.py', 'tests/test_ending_return.py', 'tsensei_v3_2_pianist/mjlab_env/ending.py', 'tsensei_v3_2_pianist/mjlab_env/preparation.py', 'tsensei_v3_2_pianist/mjlab_env/source_runtime.py', 'tsensei_v3_2_pianist/rl_torch/train.py']
changes={}
for name in sorted(overlays):
 data=(root/name).read_bytes();old=files.get(name)
 if data!=old:changes[name]={'parent_sha256':pack.sha256(old) if old else None,'sha256':pack.sha256(data)}
 files[name]=data
output.parent.mkdir(parents=True,exist_ok=True)
lineage={'parent_archive':str(parent),'parent_sha256':pack.sha256(parent.read_bytes()),'overlays':changes,
 'rule':'Pinned first-note parent retained; only enumerated ending-return sources, tests and docs overlaid; no outputs or environments',
 'physical_trace':'No actual state clipping; transient excursions disclosed separately from hover acceptance'}
(root/'docs/ending_source_lineage_20261025.json').write_bytes(pack.canonical_json(lineage))
files['docs/ending_source_lineage_20261025.json']=pack.canonical_json(lineage)
pack.write_archive(output,files,pack.build_manifest(files))
sha=pack.sha256(output.read_bytes());output.with_suffix(output.suffix+'.sha256').write_text(sha+'  '+output.name+'\n')
print(json.dumps({'archive':str(output),'sha256':sha,'overlays':len(changes)},indent=2),flush=True)
audit=pack.audit_archive(output)
print(json.dumps({'audit':'passed','file_count':audit['file_count'],'includes_m6_assets':audit['includes_m6_assets']}),flush=True)
