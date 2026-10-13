#!/usr/bin/env python3
"""Export the trainer's eligible best checkpoint, with legacy-run compatibility.

Runs with evaluation_selection use authoritative best.pt + best.json. Missing,
ineligible or inconsistent best records fail closed; logged higher scores never
override the trainer's selection. Runs predating that contract retain their
historical logged-F1 selection rule. --select-only never allocates a GPU.
"""
import argparse
from datetime import datetime
import json
import math
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
EXACT_MUSIC_F1_MIN = 'exact_original_music_micro_f1_min'


def _exact_music_score(evaluation):
    try:
        from tsensei_v3_2_pianist.rl_torch.checkpoint import exact_music_f1_min
    except ModuleNotFoundError as exc:
        if exc.name != 'torch':
            raise
        python = ROOT/'training_mjlab/.venv/bin/python'
        if not python.is_file():
            raise ValueError('Exact best validation requires training_mjlab/.venv/bin/python') from exc
        code = ('import json,sys; from tsensei_v3_2_pianist.rl_torch.checkpoint import exact_music_f1_min; '
                'print(json.dumps(exact_music_f1_min(json.load(sys.stdin))))')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', PYTHONPATH=str(ROOT))
        try:
            return json.loads(subprocess.check_output([str(python), '-c', code],
                input=json.dumps(evaluation), text=True, env=env, cwd=ROOT))
        except subprocess.CalledProcessError as error:
            raise ValueError('Could not validate original-clock best evaluation on CPU') from error
    return exact_music_f1_min(evaluation)


def read_checkpoint_metadata(checkpoint):
    """Read trusted local weights on CPU even when this Python lacks Torch."""
    try:
        import torch
    except ImportError:
        python = ROOT/'training_mjlab/.venv/bin/python'
        if not python.is_file():
            raise ValueError('Reading authoritative best requires Torch or training_mjlab/.venv/bin/python')
        code = ('import json,sys,torch; '
                'p=torch.load(sys.argv[1],map_location="cpu",weights_only=True); '
                'print(json.dumps(p["metadata"],allow_nan=False))')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        try:
            return json.loads(subprocess.check_output([str(python), '-c', code, str(checkpoint)],
                text=True, env=env))
        except subprocess.CalledProcessError as exc:
            raise ValueError('Could not read authoritative best checkpoint metadata on CPU') from exc
    try:
        return torch.load(checkpoint, map_location='cpu', weights_only=True)['metadata']
    except (KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError('Invalid authoritative best checkpoint metadata') from exc


def _validate_authoritative_best(config, metadata, best):
    if not isinstance(metadata, dict) or not isinstance(best, dict):
        raise ValueError('Authoritative metadata and best record must be objects')
    selection = config['evaluation_selection']
    if metadata.get('evaluation_selection') != selection:
        raise ValueError('best.pt selection contract differs from run config')
    state = metadata.get('evaluation_selection_state') or {}
    if state.get('config') != selection or state.get('best') != best:
        raise ValueError('best.pt selection state and best.json do not match')
    for key in ('vector_step', 'updates'):
        if type(best.get(key)) is not int or best[key] < 0 or metadata.get(key) != best[key]:
            raise ValueError('best.pt and best.json disagree on '+key)
    score = best.get('score')
    if (best.get('metric') != selection.get('best_metric')
            or best.get('metric') not in ('music_only_f1_min', EXACT_MUSIC_F1_MIN)
            or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Invalid authoritative best music criterion/score')
    if best.get('best_eligible') is False or best.get('best_ineligible_reasons'):
        raise ValueError('Trainer best record is explicitly ineligible')
    evaluation = best.get('evaluation') or {}
    evaluation_score = evaluation.get('music_only_f1_min')
    if best['metric'] == EXACT_MUSIC_F1_MIN:
        evaluation_score, reasons = _exact_music_score(evaluation)
        if reasons:
            raise ValueError('Authoritative exact best has incomplete physical diagnostics: ' + ', '.join(reasons))
    if (metadata.get('last_evaluation_vector_step') != best['vector_step']
            or metadata.get('last_evaluation') != evaluation
            or evaluation_score != score):
        raise ValueError('best.pt saved evaluation differs from best.json')
    expected = evaluation.get('evaluation_episodes')
    if (type(expected) is not int or expected <= 0
            or evaluation.get('completed_episodes') != expected
            or evaluation.get('truncated_episodes') != 0):
        raise ValueError('Authoritative best has no complete episode evaluation')
    threshold = selection.get('max_key_underside_substeps')
    if threshold is not None:
        value = evaluation.get('key_underside_substeps_max')
        hz = evaluation.get('key_contact_physics_sample_hz')
        if (best.get('best_eligible') is not True
                or best.get('max_key_underside_substeps') != threshold
                or type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or value != math.floor(value) or value > threshold
                or evaluation.get('key_contact_diagnostics_coverage') != 1.0
                or evaluation.get('key_contact_substep_coverage') != 1.0
                or evaluation.get('key_contact_diagnostics_valid') is not True
                or type(hz) not in (int, float) or not math.isfinite(hz) or hz <= 0):
            raise ValueError('Authoritative best fails configured key-underside eligibility/coverage')


def _authoritative_best(run, config, *, attempts=3):
    selection = config['evaluation_selection']
    if not isinstance(selection, dict) or selection.get('save_best') is not True:
        raise ValueError('This run has an evaluation-selection contract but best saving is disabled; no eligible best can be exported')
    checkpoint, record = run/'best.pt', run/'best.json'
    failure = None
    for attempt in range(attempts):
        try:
            # Torch checkpoints are atomically replaced; best.json can briefly
            # lag or be mid-write. Require one stable, mutually matching pair.
            before = checkpoint.stat()
            first_record = record.read_text()
            metadata = read_checkpoint_metadata(checkpoint)
            best = json.loads(first_record)
            after = checkpoint.stat()
            if ((before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_ino, after.st_size, after.st_mtime_ns)
                    or first_record != record.read_text()):
                raise ValueError('Trainer updated best while selection was being read')
            _validate_authoritative_best(config, metadata, best)
            criterion = ('trainer authoritative best: highest eligible minimum-episode '
                         + selection['best_metric'] + ', complete episodes')
            if selection.get('max_key_underside_substeps') is not None:
                criterion += '; key_underside_substeps_max <= '+str(selection['max_key_underside_substeps'])+' with complete valid physics coverage'
            return dict(run=str(run), checkpoint=str(checkpoint),
                evaluation=dict(vector_step=best['vector_step'], updates=best['updates'], eval=best['evaluation']),
                candidate_count=1, criterion=criterion, authoritative=True,
                best_record=best, evaluation_selection=selection)
        except (OSError, ValueError) as exc:
            failure = exc
            if attempt+1 < attempts: time.sleep(.05)
    raise ValueError(f'No consistent eligible authoritative best in {run}; refusing legacy/unsafe fallback: {failure}') from failure


def select_best(run):
    run = Path(run).resolve()
    config = json.loads((run/'config.json').read_text()) if (run/'config.json').is_file() else {}
    if 'evaluation_selection' in config:
        return _authoritative_best(run, config)
    summary = json.loads((run/'summary.json').read_text()) if (run/'summary.json').is_file() else {}
    lines = (run/'metrics.jsonl').read_text().splitlines()
    candidates = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines)-1:  # A running trainer may be appending this line.
                continue
            raise
        if 'eval' not in row:
            continue
        evaluation = row['eval']
        if not all(math.isfinite(float(evaluation[k])) for k in ('f1','return')):
            continue
        checkpoint = run/f"step-{row['vector_step']}.pt"
        if not checkpoint.is_file() and summary.get('status') == 'completed' and row['vector_step'] == summary.get('vector_steps'):
            checkpoint = run/'final.pt'
        if checkpoint.is_file():
            candidates.append((checkpoint,row))
    if not candidates:
        raise ValueError(f'No saved checkpoint has a finite logged evaluation in {run}')
    checkpoint, row = max(candidates, key=lambda item: (item[1]['eval']['f1'],item[1]['eval']['return'],item[1]['vector_step']))
    return dict(run=str(run),checkpoint=str(checkpoint),evaluation=row,candidate_count=len(candidates),
                criterion='highest logged mean-frame key F1; ties: return, then later vector step')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--select-only',action='store_true',help='Print selection JSON without GPU allocation or video export')
    p.add_argument('--scene',type=Path,default=None)
    p.add_argument('--task',type=Path,default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,help='New output directory; default uses a unique timestamp')
    p.add_argument('--audio',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--camera-elevation',type=float,help='Frontal elevation in degrees; default -30')
    p.add_argument('--camera-distance',type=float,help='Orthographic framing scale; default 1.05')
    args=p.parse_args()
    selection=select_best(args.run_dir)
    print(json.dumps(selection,indent=2),flush=True)
    if args.select_only:
        return
    from tsensei_v3_2_pianist.mjlab_env.paths import for_checkpoint
    args.scene,args.task=for_checkpoint(selection['checkpoint'],args.scene,args.task)
    output=args.output or ROOT/'preview/outputs/replays'/f"{args.run_dir.resolve().name}_best-{Path(selection['checkpoint']).stem}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    if output.exists():
        p.error('Output directory already exists')
    command=[sys.executable,str(Path(__file__).with_name('render_mjlab_checkpoint.py')),
             '--checkpoint',selection['checkpoint'],'--scene',str(args.scene),'--task',str(args.task),
             '--device',args.device,'--camera-view','front','--camera-framing','full',
             '--audio' if args.audio else '--no-audio','--output',str(output)]
    if args.camera_elevation is not None:
        command.extend(['--camera-elevation',str(args.camera_elevation)])
    if args.camera_distance is not None:
        command.extend(['--camera-distance',str(args.camera_distance)])
    subprocess.run(command,check=True)
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(f"Video: {(output/'replay.mp4').resolve()}",flush=True)


if __name__=='__main__':
    main()
