#!/usr/bin/env python3
"""Read coupling experiment results on CPU and write deterministic JSON/Markdown.

No GPU imports, training, monitoring loop or checkpoint deserialization. Relative
output_prefix paths resolve against the manifest directory. Optional --output is
a basename (or .json/.md path); both report extensions are written beside it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


CORE_METRICS = (
    'music_only_f1', 'music_only_f1_min', 'music_only_micro_f1', 'music_only_micro_f1_min',
    'music_only_substep_key_micro_f1', 'music_only_substep_key_micro_f1_min',
    'music_only_wrong_key_substeps', 'music_only_wrong_key_substeps_max',
    'hand_collision_substeps', 'hand_collision_substeps_max',
    'self_collision_substeps', 'self_collision_substeps_max',
    'inter_hand_collision_substeps', 'inter_hand_collision_substeps_max',
    'key_underside_substeps', 'key_underside_substeps_max',
    'key_side_or_under_substeps', 'key_side_or_under_substeps_max',
    'collision_diagnostics_valid', 'collision_diagnostics_coverage', 'collision_substep_coverage',
    'key_contact_diagnostics_valid', 'key_contact_diagnostics_coverage', 'key_contact_substep_coverage',
    'substep_key_diagnostics_valid', 'substep_key_diagnostics_coverage', 'substep_key_sample_hz',
    'completed_episodes', 'truncated_episodes', 'evaluation_episodes', 'evaluation_num_envs',
)
FAILED_STATES = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED'}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def slurm_states(job_ids):
    """Single read-only accounting snapshot; absence never implies running."""
    ids = sorted({str(job) for job in job_ids if re.fullmatch(r'\d+', str(job))})
    if not ids:
        return {}
    try:
        result = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(ids),
                                 '--format=JobIDRaw,State'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode:
        return {}
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) >= 2 and fields[0] in ids:
            states[fields[0]] = fields[1].split()[0].rstrip('+')
    return states


def read_json(path, issues):
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected object')
        return value
    except (OSError, ValueError) as error:
        issues.append(f'{path.name}: {error}')
        return {}


def read_metrics(path, issues):
    if not path.is_file():
        return []
    rows = []
    try:
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('expected object')
                except ValueError:
                    # Training may be in the middle of its last write. Never
                    # turn a partial record into zero counters or final F1.
                    issues.append(f'metrics.jsonl line {number}: incomplete or invalid JSON; excluded')
                    continue
                rows.append(row)
    except OSError as error:
        issues.append(f'metrics.jsonl: {error}')
    return rows


def metric_subset(evaluation):
    if not isinstance(evaluation, dict):
        return None
    return {key: value for key, value in evaluation.items()
            if key in CORE_METRICS or key.startswith('finger_coupling_')}


def resolve_run(prefix):
    """Match timestamped outputs exactly, so similarly named arms never mix."""
    pattern = re.compile(re.escape(prefix.name) + r'-\d{8}-\d{6}-\d{6}(?:-\d+)?$')
    return sorted(path for path in prefix.parent.iterdir()
                  if path.is_dir() and pattern.fullmatch(path.name)) if prefix.parent.is_dir() else []


def run_status(summary, job_state, found):
    if summary.get('status') == 'completed':
        return 'completed', 'summary.json'
    if job_state in FAILED_STATES:
        return 'failed', 'Slurm accounting'
    if job_state in {'PENDING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD'}:
        return 'pending', 'Slurm accounting'
    if job_state in {'RUNNING', 'CONFIGURING', 'COMPLETING', 'RESIZING'}:
        return 'running', 'Slurm accounting'
    if job_state == 'COMPLETED':
        return 'incomplete', 'Slurm completed but training summary is absent'
    return ('incomplete' if found else 'unknown'), 'No training summary or current Slurm state'


def summarize_run(spec, manifest, directory, states):
    prefix = Path(spec['output_prefix']).expanduser()
    if not prefix.is_absolute():
        prefix = directory / prefix
    prefix = prefix.resolve()
    result = dict(name=spec['name'], job_id=spec['job_id'], output_prefix=str(prefix),
                  expected_action_size=spec['expected_action_size'],
                  expected_observation_size=spec['expected_observation_size'],
                  slurm_state=states.get(str(spec['job_id'])), data_issues=[])
    issues = result['data_issues']
    paths = resolve_run(prefix)
    if len(paths) > 1:
        result.update(status='ambiguous', status_basis='Multiple timestamped output directories',
                      candidate_directories=[str(path) for path in paths], budget_complete=False,
                      contract_matches=False)
        return result
    run = paths[0] if paths else None
    result['run_directory'] = str(run) if run else None
    summary = read_json(run/'summary.json', issues) if run else {}
    config = read_json(run/'config.json', issues) if run else {}
    records = read_metrics(run/'metrics.jsonl', issues) if run else []
    best = read_json(run/'best.json', issues) if run else {}
    if not best:
        best = (summary.get('evaluation_selection_state') or {}).get('best') or {}
    if not best:
        improved = [row for row in records if row.get('checkpoint_selection', {}).get('improved')]
        if improved:
            last = improved[-1]
            best = dict(vector_step=last.get('vector_step'), updates=last.get('updates'),
                        evaluation=last.get('eval'))
    status, basis = run_status(summary, result['slurm_state'], bool(run))
    result.update(status=status, status_basis=basis)
    observed = max(records, key=lambda row: (row.get('transitions', -1), row.get('updates', -1)), default={})
    result['transitions'] = summary.get('transitions', observed.get('transitions'))
    result['updates_this_run'] = summary.get('updates_this_run')
    result['updates_total'] = summary.get('updates_total', observed.get('updates'))
    result['progress_source'] = 'summary.json' if summary else 'latest complete metrics.jsonl record'
    result['observed_vector_step'] = summary.get('vector_steps', observed.get('vector_step'))
    result['replay_insertions'] = summary.get('replay_insertions')
    result['action_size'] = config.get('action_size')
    result['observation_size'] = config.get('observation_size')
    result['target_entropy'] = config.get('target_entropy')
    result['contract_matches'] = (bool(config) and config.get('action_size') == spec['expected_action_size']
                                  and config.get('observation_size') == spec['expected_observation_size'])
    expected_updates = manifest.get('expected_updates_per_arm', manifest.get('expected_updates', 726656))
    result['budget_complete'] = (status == 'completed'
        and result['transitions'] == manifest['requested_samples_per_arm']
        and result['replay_insertions'] == manifest['requested_samples_per_arm']
        and result['updates_this_run'] == expected_updates)
    result['best'] = (dict(vector_step=best.get('vector_step'), updates=best.get('updates'),
                           metrics=metric_subset(best.get('evaluation'))) if best else None)
    result['final'] = metric_subset(summary.get('final_evaluation'))
    evaluations = [row for row in records if isinstance(row.get('eval'), dict)]
    result['latest_evaluation'] = (dict(transitions=evaluations[-1].get('transitions'),
        metrics=metric_subset(evaluations[-1]['eval'])) if evaluations else None)
    start = max(0, manifest['requested_samples_per_arm']-500000)
    end = manifest['requested_samples_per_arm']
    # Fixed end-of-budget window, not the latest available 500k while running.
    window = [row for row in evaluations if finite(row.get('transitions'))
              and start <= row['transitions'] <= end and not row.get('baseline', False)]
    values = [row['eval'].get('music_only_f1') for row in window]
    minimums = [row['eval'].get('music_only_f1_min') for row in window]
    result['last_500k'] = dict(start_transition=start, end_transition=end, evaluation_count=len(window),
        observed_start_transition=window[0]['transitions'] if window else None,
        observed_end_transition=window[-1]['transitions'] if window else None,
        mean_music_f1=statistics.mean(values) if values and all(map(finite, values)) else None,
        mean_episode_min_music_f1=statistics.mean(minimums) if minimums and all(map(finite, minimums)) else None,
        window_complete=bool(result['budget_complete'] and window),
        averaging='unweighted mean across recorded evaluations in fixed final 500000 transitions')
    result['elapsed_seconds'] = summary.get('elapsed_s')
    result['evaluation_timing'] = summary.get('evaluation_timing')
    return result


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or not isinstance(manifest.get('runs'), list) or not manifest['runs']:
        raise ValueError('Manifest must contain a nonempty runs list')
    if type(manifest.get('requested_samples_per_arm')) is not int or manifest['requested_samples_per_arm'] <= 0:
        raise ValueError('requested_samples_per_arm must be a positive integer')
    names = set()
    for spec in manifest['runs']:
        for key in ('name', 'job_id', 'output_prefix', 'expected_action_size', 'expected_observation_size'):
            if key not in spec:
                raise ValueError('Run is missing '+key)
        if not isinstance(spec['name'], str) or spec['name'] in names:
            raise ValueError('Run names must be unique strings')
        names.add(spec['name'])
        if not isinstance(spec['output_prefix'], str) or not spec['output_prefix']:
            raise ValueError('output_prefix must be a nonempty string')
        for key in ('expected_action_size', 'expected_observation_size'):
            if type(spec[key]) is not int or spec[key] <= 0:
                raise ValueError(key+' must be a positive integer')


def summarize(manifest, directory, states=None):
    validate_manifest(manifest)
    if states is None:
        states = slurm_states(spec['job_id'] for spec in manifest['runs'])
    runs = [summarize_run(spec, manifest, directory, states) for spec in manifest['runs']]
    complete = all(row['budget_complete'] and row['contract_matches'] and not row['data_issues'] for row in runs)
    return dict(format='coupling_comparison_current', requested_samples_per_arm=manifest['requested_samples_per_arm'],
                expected_updates_per_arm=manifest.get('expected_updates_per_arm', manifest.get('expected_updates', 726656)),
                comparison_complete=complete,
                conclusion=('complete_results_ready_for_review' if complete else 'incomplete_no_winner'),
                winner=None, runs=runs,
                notes=['No F1 winner is inferred from incomplete runs.',
                       'Completed budgets alone do not establish statistical significance or physical safety.',
                       'Angle statistics are control-endpoint/current-assignment diagnostics, not 200Hz contact-angle measurements.',
                       'Running progress uses the latest complete log record, not unrecorded live counters.'])


def number(value, digits=5):
    return f'{value:.{digits}f}' if finite(value) else '—'


def markdown(report):
    lines = ['# 手指耦合训练对照', '',
             ('三组完整预算和维度校验已通过，可检查结果；此报告不自动选定方案。'
              if report['comparison_complete'] else '对照尚未完整完成或校验未通过，暂不判断 F1 优胜组。'), '',
             '| 组别 | 作业 | 状态 | 已记录样本 | 更新数 | 动作/观测 | best F1最低/平均 | final音乐F1 | final 200Hz F1 | 末50万均值 |',
             '| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |']
    for run in report['runs']:
        best = (run.get('best') or {}).get('metrics') or {}
        final = run.get('final') or {}
        window = run.get('last_500k') or {}
        updates = run.get('updates_this_run')
        if updates is None:
            updates = run.get('updates_total')
        name = str(run['name']).replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {name} | {run['job_id']} | {run['status']} | {run.get('transitions', '—')} | {updates if updates is not None else '—'} | "
                     f"{run.get('action_size', '—')}/{run.get('observation_size', '—')} | "
                     f"{number(best.get('music_only_f1_min'))}/{number(best.get('music_only_f1'))} | "
                     f"{number(final.get('music_only_f1'))} | {number(final.get('music_only_substep_key_micro_f1'))} | "
                     f"{number(window.get('mean_music_f1'))}{'（未完整）' if not window.get('window_complete') else ''} |")
    lines += ['', '末50万使用固定计划预算末段的评估均值；训练中的记录只代表已保存的进度。', '',
              '## 物理与角度诊断', '',
              '| 组别 | 数据来源 | 手碰撞子步/回合 | 键底子步/回合 | 当前用指实际角误差均值/最大（°） |',
              '| --- | --- | ---: | ---: | --- |']
    for run in report['runs']:
        metrics = run.get('final') or (run.get('latest_evaluation') or {}).get('metrics') or {}
        source = 'final' if run.get('final') else '最近评估'
        name = str(run['name']).replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {name} | {source} | {number(metrics.get('hand_collision_substeps'), 2)} | "
                     f"{number(metrics.get('key_underside_substeps'), 2)} | "
                     f"{number(metrics.get('finger_coupling_active_actual_error_deg_mean'), 2)}/"
                     f"{number(metrics.get('finger_coupling_active_actual_error_deg_max'), 2)} |")
    lines += ['', '角度为20Hz控制周期末/当前谱面用指统计，不能代表每次200Hz实际接触瞬间。']
    for run in report['runs']:
        if run.get('data_issues'):
            lines += ['', f"数据问题（{run['name']}）："+ '; '.join(run['data_issues'])]
        if run.get('status') == 'ambiguous':
            lines += ['', f"{run['name']} 有多个匹配输出目录，未选择其中任何一个。"]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Report basename; writes .json and .md')
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    try:
        report = summarize(json.loads(manifest_path.read_text()), manifest_path.parent)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output = args.output or manifest_path.with_name(manifest_path.stem+'_summary')
    if output.suffix in ('.json', '.md'):
        output = output.with_suffix('')
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = Path(str(output)+'.json'), Path(str(output)+'.md')
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')
    md_path.write_text(markdown(report))
    print(md_path.read_text(), end='')
    print(f'JSON: {json_path.resolve()}\nMarkdown: {md_path.resolve()}')


if __name__ == '__main__':
    main()
