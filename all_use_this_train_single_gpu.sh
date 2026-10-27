#!/usr/bin/env bash
# 单卡入口：动态复用双卡入口的全部参数和冻结源码，不复制训练默认值。
# 用法：bash all_use_this_train_single_gpu.sh [原入口选项及训练参数...]
# --print-script：仅输出展开脚本，供检查，不申请资源。
set -euo pipefail
SINGLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 - "$SINGLE_ROOT" "$@" <<'PY'
import os
from pathlib import Path
import shlex
import sys

root = Path(sys.argv[1])
args = sys.argv[2:]
source = root / 'all_use_this_train.sh'
script = source.read_text()
# Fail closed if the shared entry changes structure; never silently request two GPUs.
replacements = {
    'ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"':
        'ROOT=' + shlex.quote(str(root)),
    '"$ROOT/scripts/hpc/submit.sh" --gpus 2':
        '"$ROOT/scripts/hpc/submit.sh" --gpus 1',
    '--device cuda:0 --collector-devices cuda:0 --learner-device cuda:1 --collector-pipeline':
        '--device cuda:0 --collector-devices cuda:0 --learner-device cuda:0 --no-collector-pipeline',
    '"$@"\n': '"$@" --device cuda:0 --collector-devices cuda:0 --learner-device cuda:0 --no-collector-pipeline\n',
}
# The final argument override must be scoped to the final command, not arm validation.
for old, new in list(replacements.items())[:-1]:
    if script.count(old) != 1:
        sys.exit(f'Single-GPU entry: shared entry changed; expected exactly one {old!r}')
    script = script.replace(old, new)
if not script.endswith('"$@"\n'):
    sys.exit('Single-GPU entry: shared entry final argument layout changed')
script = script[:-len('"$@"\n')] + replacements['"$@"\n']
if args[:1] == ['--print-script']:
    print(script, end='')
    sys.exit(0)
os.chdir(root)
os.execvp('bash', ['bash', '-c', script, str(source), *args])
PY
