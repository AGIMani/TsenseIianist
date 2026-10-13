#!/usr/bin/env python3
"""Summarize a benchmark_training_speed.py run using existing train metrics."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import statistics


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return dict(count=len(values), median=statistics.median(values),
                mean=statistics.mean(values), minimum=values[0], maximum=values[-1],
                p10=values[round((len(values)-1)*.1)],
                p90=values[round((len(values)-1)*.9)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('--burnin-vector-steps', type=int, default=40,
                        help='Exclude early log intervals; default40 for the512-environment baseline')
    args = parser.parse_args()
    out = args.benchmark.resolve()
    invocation = json.loads((out/'invocation.json').read_text())
    run = Path(invocation['run_dir'])
    summary = json.loads((run/'summary.json').read_text())
    config = json.loads((run/'config.json').read_text())
    records = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    rows = [row for row in records if 'loop_transitions_per_second' in row]
    evaluations = {row['vector_step'] for row in records if 'eval' in row}
    resets = {row['vector_step'] for row in records if 'episodes' in row}
    intervals = []
    for previous, current in zip(rows, rows[1:]):
        start, end = previous['vector_step'], current['vector_step']
        elapsed = (current['transitions']/current['loop_transitions_per_second'] -
                   previous['transitions']/previous['loop_transitions_per_second'])
        intervals.append(dict(start=start, end=end, seconds=elapsed,
                              seconds_per_vector_step=elapsed/(end-start),
                              samples_per_second=(current['transitions']-previous['transitions'])/elapsed,
                              includes_evaluation=any(start <= step < end for step in evaluations),
                              includes_reset=any(start < step <= end for step in resets)))
    normal = [row for row in intervals if row['start'] >= args.burnin_vector_steps and
              not row['includes_evaluation'] and not row['includes_reset']]
    regular = stats([row['seconds_per_vector_step'] for row in normal])
    assert regular, 'At least two stable non-evaluation log intervals are required'
    evaluation_overheads = [row['seconds']-regular['median']*(row['end']-row['start'])
                            for row in intervals if row['start'] >= args.burnin_vector_steps and row['includes_evaluation']]
    with (out/'gpu.csv').open() as stream:
        gpu = [{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(stream)]
    first = datetime.datetime.strptime(gpu[0]['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
    # Time-based filter excludes source rebuild/initial compilation, not low-util samples.
    active = [row for row in gpu if
              (datetime.datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f')-first).total_seconds() >= 20]
    telemetry = {key: stats([float(row[key]) for row in active]) for key in
                 ['utilization.gpu [%]', 'utilization.memory [%]', 'memory.used [MiB]',
                  'power.draw [W]', 'temperature.gpu', 'clocks.current.sm [MHz]']
                 if active and key in active[0]}
    result = dict(run_dir=str(run), training_summary=summary,
                  trainer_total_samples_per_second=summary['transitions']/summary['elapsed_s'],
                  process_total_samples_per_second=summary['transitions']/invocation['wall_seconds'],
                  stable_seconds_per_vector_step=regular,
                  stable_samples_per_second=config['num_envs']/regular['median'],
                  stable_interval_samples_per_second=stats([row['samples_per_second'] for row in normal]),
                  evaluation_count=len(evaluations), reset_steps=sorted(resets),
                  evaluation_extra_seconds=stats(evaluation_overheads),
                  gpu_telemetry_after_first_20_seconds=telemetry, intervals=intervals,
                  burnin_vector_steps=args.burnin_vector_steps,
                  explanation='Stable intervals after burnin exclude evaluation/reset; total throughput includes them. GPU telemetry still includes evaluations.')
    (out/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='intervals'}, indent=2))


if __name__ == '__main__':
    main()
