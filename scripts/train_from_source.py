#!/usr/bin/env python3
# 2026-10-25: External initial positions use world-frame metres; yaw is the palm-base joint angle in radians.
"""Train using current source + explicit parameters; write complete config.json.

No persistent frozen scene or source_cache is read or created. Composer builds
an ephemeral compiler workspace, then the MJLab runner uses its native model.
Replay rebuilds that model from the saved source recipe.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def timestamped_output(rest, parser):
    """Choose a fresh run directory in Beijing time, without touching old runs."""
    requested = Path('training_mjlab/outputs/sac')
    for index, item in enumerate(rest):
        if item == '--output-dir':
            if index + 1 == len(rest) or rest[index + 1].startswith('--'):
                parser.error('--output-dir requires a directory path')
            requested = Path(rest[index + 1])
        elif item.startswith('--output-dir='):
            requested = Path(item.partition('=')[2])
    if not requested.name:
        parser.error('--output-dir must name a run directory')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    output = requested.with_name(f'{requested.name}-{stamp}')
    serial = 1
    while output.exists():
        output = requested.with_name(f'{requested.name}-{stamp}-{serial:02d}')
        serial += 1
    return output


def resolve_exact_note_timing(cli_value, rest):
    """The exporter and learner must resolve the same CLI-over-JSON switch."""
    if cli_value is not None:
        return cli_value
    reward_path = None
    for index, argument in enumerate(rest):
        if argument == '--reward-config' and index + 1 < len(rest):
            reward_path = Path(rest[index + 1])
        elif argument.startswith('--reward-config='):
            reward_path = Path(argument.partition('=')[2])
    value = (json.loads(reward_path.read_text()).get('performance', {}).get('exact_note_timing', True)
             if reward_path is not None else True)
    if type(value) is not bool:
        raise ValueError('performance.exact_note_timing must be true/false')
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    from tsensei_v3_2_pianist.initial_world_pose import (
        add_arguments as add_pose_arguments, reject_legacy_pose_arguments)
    add_pose_arguments(p)
    p.add_argument('--wrist-pitch-range-scale',type=float,default=1.,
                   help='Scale both physical Pitch limits and policy target bounds; (0,1], default unchanged')
    p.add_argument('--control-hz',type=float,default=20.)
    p.add_argument('--physics-hz',type=float,default=200.)
    p.add_argument('--midi',default='TwinkleTwinkleRousseau')
    p.add_argument('--lookahead',type=int)
    p.add_argument('--frame-stack',type=int,default=1)
    p.add_argument('--initial-buffer-time',type=float,default=0.)
    p.add_argument('--release-tail-time',type=float,default=0.,
                   help='Real silent training tail in seconds; must contain whole control frames')
    p.add_argument('--score-validation',choices=('legacy','strict_v1'),default='legacy',
                   help='strict_v1 rejects sampling loss; legacy records exact-score coverage without rejecting it')
    p.add_argument('--exact-note-timing',action=argparse.BooleanOptionalAction,default=None,
                   help='Preserve original MIDI times for physical-substep key reward, observations and evaluation')
    p.add_argument('--ending-time', type=float, default=0., help='Minimum physical return to original pose after full playback; outside SAC')
    p.add_argument('--preparation-time',type=float,default=0.,
                   help='Preparation main-motion seconds; 0 disables preparation')
    p.add_argument('--preparation-mode',choices=('fixed_v1','policy_v1','first_note_v1'),default='fixed_v1',
                   help='first_note_v1 hovers before learning; policy_v1 trains during preparation; fixed_v1 descends')
    p.add_argument('--preparation-lift',type=float,default=.02,
                   help='World-Z initial lift in metres; does not create a Palm_Base_Tz action')
    p.add_argument('--gravity-compensation',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--action-reward-observation',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--disable-forearm-reward',action='store_true')
    p.add_argument('--task-profile', choices=('tsensei_v1','tsensei_collision_v1'), default='tsensei_v1')
    p.add_argument('--collision-profile', choices=('legacy_mesh_v1','convex_hull_v1','shadow_primitive_v1'),
                   default='convex_hull_v1')
    p.add_argument('--drive-profile', choices=('legacy_v1','shadow_v1'), default='legacy_v1')
    p.add_argument('--progress',action=argparse.BooleanOptionalAction,default=True,
                   help='Show startup stages and training progress, throughput and ETA (default: on)')
    p.add_argument('--build-only',action='store_true',help='Validate construction; temporary workspace is deleted on exit')
    from swanlab_upload import add_arguments, run_with_upload
    add_arguments(p)
    reject_legacy_pose_arguments(sys.argv[1:] if argv is None else argv, p)
    a,rest=p.parse_known_args(argv)
    if any(x.split("=")[0] in ("--thumb-target-mode", "--thumb-white-target-distance", "--thumb-black-target-distance") for x in rest):
        p.error("Thumb-only target options were removed; all fingers use white 40 mm / black 20 mm")
    from tsensei_v3_2_pianist.mjlab_env.arm_options import preflight as arm_preflight
    arm_preflight(rest)
    if any(x.split('=')[0] in ('--scene','--task','--source-config','--cache-dir') for x in rest):
        p.error('Source entrypoint reconstructs inputs; scene/task/cache paths are not accepted')
    output = None if a.build_only or '--inspect-only' in rest else timestamped_output(rest,p)
    if output is not None:
        print(f'本次输出目录（北京时间）：{output}',file=sys.stderr,flush=True)
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
    fields=vars(a).copy();fields.pop('build_only');fields.pop('progress')
    fields['exact_note_timing'] = resolve_exact_note_timing(a.exact_note_timing,rest)
    fields = {key: value for key, value in fields.items()
              if key != 'swanlab' and not key.startswith('swanlab_')}
    if a.progress:
        print('[构建 1/2] 读取源码与完整训练配置...',file=sys.stderr,flush=True)
    recipe=make_recipe(**fields)
    if a.progress:
        print('[构建 2/2] 创建 MuJoCo 模型与乐谱...',file=sys.stderr,flush=True)
    scene,task,config=build(recipe)
    if a.build_only:
        print(json.dumps({'pose':recipe['pose'],'environment':recipe['environment'],'status':'built_from_source'},indent=2));return
    from source_pythonpath import source_pythonpath
    env=os.environ.copy();env['PYTHONPATH']=source_pythonpath(ROOT)
    for name in ('TSENSEI_V32_EXPORT_POSE','TSENSEI_V32_COLLISION_PROFILE','TSENSEI_V32_DRIVE_PROFILE'):
        env.pop(name,None)
    if a.progress:
        print('环境构建完成，开始 GPU 初始化。首次编译完成后显示训练进度、速度及预计剩余时间。',
              file=sys.stderr,flush=True)
    command=[str(ROOT/'training_mjlab/.venv/bin/python'),'-u','-m','tsensei_v3_2_pianist.rl_torch.train',
             '--scene',str(scene),'--task',str(task),'--source-config',str(config),
             '--preparation-time',str(a.preparation_time),'--preparation-lift',str(a.preparation_lift),
             '--preparation-mode',a.preparation_mode,'--ending-time',str(a.ending_time),
             '--progress' if a.progress else '--no-progress',
             *([] if a.exact_note_timing is None else
               ['--exact-note-timing' if a.exact_note_timing else '--no-exact-note-timing']),*rest]
    if output is not None:
        command.extend(('--output-dir',str(output)))
    run_with_upload(command, root=ROOT, env=env, output=output, options=a)

if __name__=='__main__':main()
