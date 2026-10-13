#!/usr/bin/env python3
"""Archive visible assistant answers from this local thread and its ancestors.

Only assistant commentary/final messages are exported. Reasoning, tool output,
system instructions and unrelated threads are excluded. Missing history is
reported explicitly. Run again after the final answer to include that answer.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def collect(base, thread, cutoff=None, visited=None):
    visited = set() if visited is None else visited
    if thread in visited:
        raise ValueError('Cycle in thread ancestry')
    visited.add(thread)
    paths = list(base.rglob('*'+thread+'.jsonl'))
    if len(paths) != 1:
        return [], [f'会话 {thread} 找到 {len(paths)} 份记录，无法唯一读取。']
    rows = []
    warnings = []
    for line in paths[0].read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.append(f'会话 {thread} 的一行记录不完整，已跳过。')
    meta = next(r['payload'] for r in rows if r['type'] == 'session_meta')
    answers = []
    if meta.get('forked_from_id'):
        prior, issues = collect(base, meta['forked_from_id'], meta.get('forked_from_ordinal_exclusive'), visited)
        answers.extend(prior); warnings.extend(issues)
    ordinals = [r['ordinal'] for r in rows if 'ordinal' in r]
    if cutoff is not None and ordinals and max(ordinals) < cutoff-1:
        warnings.append(f'父会话 {thread} 本地仅保存到序号 {max(ordinals)}；分支引用截止序号 {cutoff}，中间历史未在此文件中找到。')
    for row in rows:
        if cutoff is not None and row.get('ordinal', cutoff) >= cutoff:
            continue
        p = row['payload']
        if row['type'] != 'response_item' or p.get('type') != 'message' or p.get('role') != 'assistant':
            continue
        phase = p.get('phase', p.get('channel'))
        if phase not in ('commentary', 'final_answer', 'final'):
            continue
        text = '\n'.join(c.get('text','') for c in p.get('content', []) if c.get('type') in ('output_text', 'text'))
        if text:
            answers.append(dict(timestamp=row['timestamp'], thread=thread, ordinal=row.get('ordinal'), phase=phase, text=text))
    return answers, warnings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thread', default=os.environ.get('CODEX_THREAD_ID'))
    p.add_argument('--sessions', type=Path, default=Path.home()/'.codex/sessions')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.thread:
        p.error('--thread is required outside the current Codex session')
    answers, warnings = collect(a.sessions, a.thread)
    finals = sum(row['phase'] in ('final_answer', 'final') for row in answers)
    lines = ['# 侧边聊天回答记录', '',
             f'导出时间（UTC）：{datetime.now(timezone.utc).isoformat()}', '',
             f'当前会话：`{a.thread}`。已找到 {len(answers)} 条回答，其中 {finals} 条最终回答，其余为进度说明。', '',
             '范围：当前会话及其父会话在本机实际保存的助手可见文字，逐条保留原文；不含内部推理、工具结果或其他会话。历史回答可能已被后续方案替代，请按时间阅读。', '',
             '本文件是导出时点的快照，之后的回答可再次运行脚本追加到重新生成的文档中。', '', '# 历史完整性', '']
    lines.extend(['- '+s for s in warnings] or ['本机记录未检测到上述分支序号缺口；仍不代表界面服务端记录已全部保存在本机。'])
    for index, row in enumerate(answers, 1):
        kind = '最终回答' if row['phase'] in ('final_answer', 'final') else '进度说明'
        lines.extend(['', f'## {index}. {row["timestamp"]} · {kind}', '',
                      f'来源：`{row["thread"]}`／序号 {row["ordinal"]}', '', row['text']])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(answers=len(answers), final_answers=finals, missing_history=warnings, output=str(a.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
