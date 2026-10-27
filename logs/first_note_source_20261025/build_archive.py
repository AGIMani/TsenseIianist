"""Freeze parent archive plus the explicit, reviewed preparation overlay."""
from pathlib import Path
import sys,tarfile,json,hashlib,importlib.util
root=Path(sys.argv[1]);parent=Path(sys.argv[2]);output=Path(sys.argv[3])
spec=importlib.util.spec_from_file_location('pack',root/'scripts/hpc/package_a32.py');pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
files={}
with tarfile.open(parent) as archive:
 for item in archive:
  if item.isfile() and item.name!='BUNDLE_MANIFEST.json':files[item.name]=archive.extractfile(item).read()
overlays=['AGENTS.md','preview/scripts/export_video_job.py','preview/scripts/joint_motion_plots.py','scripts/export_videos.py','tests/test_joint_motion_plots.py','tests/test_video_export.py','README.md','docs/change_log.md','docs/reward_defaults_20261006.md',
'docs/superpowers/specs/2026-10-25-first-note-preparation-design.md','docs/superpowers/plans/2026-10-25-first-note-preparation.md',
'preview/scripts/render_mjlab_checkpoint.py','preview/scripts/render_first_note.py',
'scripts/hpc/job.sbatch','scripts/hpc/submit.sh','scripts/hpc/prepare_first_note.py','scripts/train_from_source.py',
*['tsensei_v3_2_pianist/mjlab_env/'+n+'.py' for n in ['bundle','preparation','first_note_preparation','scene','source_runtime','task']],
 'tsensei_v3_2_pianist/rl_torch/train.py','tests/test_first_note_gpu.py','tests/test_first_note_preparation.py']
overlays += [p.relative_to(root).as_posix() for p in (root/'training_mjlab/inputs/turkish_march_advanced/a32_contact_refined_20261024').rglob('*') if p.is_file()]
changes={}
for name in sorted(overlays):
 data=(root/name).read_bytes();old=files.get(name)
 if data!=old:changes[name]={'parent_sha256':pack.sha256(old) if old else None,'sha256':pack.sha256(data)}
 files[name]=data
output.parent.mkdir(parents=True,exist_ok=True)
lineage={'parent_archive':str(parent),'parent_sha256':pack.sha256(parent.read_bytes()),'overlays':changes,
 'rule':'Original pinned parent membership retained, only enumerated preparation sources/docs, validated raw-physics export fixes and current default score added; no outputs or environments',
 'physical_trace':'No actual state clipping; transient excursions disclosed separately from hover acceptance'}
(root/'docs/first_note_source_lineage_20261025.json').write_bytes(pack.canonical_json(lineage))
files['docs/first_note_source_lineage_20261025.json']=pack.canonical_json(lineage)
pack.write_archive(output,files,pack.build_manifest(files))
sha=pack.sha256(output.read_bytes());output.with_suffix(output.suffix+'.sha256').write_text(sha+'  '+output.name+'\n')
print(json.dumps({'archive':str(output),'sha256':sha,'overlays':len(changes)},indent=2),flush=True)
audit=pack.audit_archive(output)
print(json.dumps({'audit':'passed','file_count':audit['file_count'],'includes_m6_assets':audit['includes_m6_assets']}),flush=True)
