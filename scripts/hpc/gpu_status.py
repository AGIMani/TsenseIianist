#!/usr/bin/env python3
"""Read-only Slurm GPU inventory for configured SSH hosts; no CUDA initialization."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import re
import subprocess
import sys


def field(text, name, default=''):
    match = re.search(r'(?<!\S)' + re.escape(name) + r'=(\S+)', text)
    return match.group(1) if match else default


def indices(text):
    result = []
    for item in text.split(','):
        if re.fullmatch(r'\d+-\d+', item):
            a, b = map(int, item.split('-'))
            result.extend(range(a, b + 1))
        elif item.isdigit():
            result.append(int(item))
        else:
            raise ValueError('Unsupported GPU index expression: ' + text)
    return result


def parse(nodes_text, jobs_text):
    nodes, jobs = [], []
    for block in re.split(r'(?=NodeName=)', nodes_text):
        name = field(block, 'NodeName')
        if not name:
            continue
        gres = field(block, 'Gres')
        matches = re.findall(r'(?:^|,)gpu(?::([^:,()]+))?:(\d+)', gres)
        total = sum(int(n) for _, n in matches)
        if not total:
            continue
        alloc = re.search(r'(?:^|,)gres/gpu=(\d+)', field(block, 'AllocTRES'))
        nodes.append(dict(node=name, model='/'.join(t or 'gpu' for t, _ in matches),
                          total=total, allocated=int(alloc.group(1)) if alloc else 0,
                          state=field(block, 'State'), owners={}))
    by_name = {n['node']: n for n in nodes}
    for block in re.split(r'(?<!\S)(?=JobId=)', jobs_text):
        jid = field(block, 'JobId')
        if not jid:
            continue
        array, task = field(block, 'ArrayJobId'), field(block, 'ArrayTaskId')
        display_id = array + '_' + task if array and task else jid
        job = dict(job_id=display_id, user=field(block, 'UserId').split('(')[0],
                   state=field(block, 'JobState'), name=field(block, 'JobName'),
                   reason=field(block, 'Reason'), requested=field(block, 'TresPerNode'))
        jobs.append(job)
        for line in block.splitlines():
            node = field(line, 'Nodes')
            if node not in by_name:
                continue
            for match in re.finditer(r'gpu[^\s()]*\(IDX:([^)]*)\)', line):
                for gpu in indices(match.group(1)):
                    by_name[node]['owners'].setdefault(gpu, []).append(job)
    for n in nodes:
        # Missing/hidden allocation details must never become "free GPUs".
        n['mapping_complete'] = len(n['owners']) == n['allocated']
        n['schedulable_state'] = not any(x in n['state'] for x in
            ('DOWN', 'DRAIN', 'FAIL', 'MAINT', 'RESERVED', 'UNKNOWN', 'NOT_RESPONDING'))
    return nodes, jobs


def query(host):
    remote = "python3 -c 'import json,subprocess; print(json.dumps({k:subprocess.check_output(v,text=True) for k,v in [(\"nodes\",[\"scontrol\",\"show\",\"nodes\"]),(\"jobs\",[\"scontrol\",\"-d\",\"show\",\"jobs\"])]}))'"
    try:
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                            host, remote], text=True, capture_output=True, timeout=45)
        if p.returncode:
            raise RuntimeError(p.stderr.strip() or 'SSH query failed')
        raw = json.loads(p.stdout)
        nodes, jobs = parse(raw['nodes'], raw['jobs'])
        return dict(host=host, nodes=nodes, jobs=jobs)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return dict(host=host, error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hosts', nargs='+', default=['volc-a32', 'h100'], help='SSH aliases')
    parser.add_argument('--json', action='store_true', help='Machine-readable output')
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=min(8, len(args.hosts))) as pool:
        reports = list(pool.map(query, args.hosts))
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    if args.json:
        print(json.dumps(dict(queried_at=stamp, hosts=reports), ensure_ascii=False, indent=2))
    else:
        print('查询时间:', stamp)
        for r in reports:
            print('\n主机:', r['host'])
            if 'error' in r:
                print('查询失败（状态未知）:', r['error']); continue
            for n in r['nodes']:
                print(f"{n['node']} {n['model']} — 已分配 {n['allocated']}/{n['total']}，节点 {n['state']}")
                for gpu in range(n['total']):
                    owners = n['owners'].get(gpu, [])
                    if owners:
                        state = '；'.join(f"{j['user']} 作业{j['job_id']} {j['state']} {j['name']}" for j in owners)
                    elif not n['mapping_complete']:
                        state = '未知：分配明细不完整'
                    elif not n['schedulable_state']:
                        state = '未分配，但节点不可调度'
                    else:
                        state = '未分配，可申请（是否立即运行由Slurm调度）'
                    print(f'  卡 {gpu}: {state}')
            pending = [j for j in r['jobs'] if j['state'] == 'PENDING']
            print(f'排队作业: {len(pending)}')
        print('\n卡号为Slurm GRES索引；空闲不等于已获授权，训练仍须通过Slurm申请。')
    return int(any('error' in r for r in reports))


if __name__ == '__main__':
    sys.exit(main())
