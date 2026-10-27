#!/usr/bin/env python3
"""Validate an export request, freeze its inputs, and submit one Slurm job.

No MuJoCo, Torch, or GPU initialization happens in this launcher.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'overview', 'hands')
ALIASES = {'forhead': 'front', 'forehead': 'front', 'backend': 'back', 'performer': 'back'}
PLOTS = ('arm-joints', 'tcp', 'hand-joints')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError('Expected a finite number')
    return value


def video_spec(tokens, number, layout):
    if not tokens or tokens[0] not in ('arms', 'noarms', 'native'):
        raise ValueError(f'--video{number} must begin with arms, noarms or native')
    views, options = [], {}
    for token in tokens[1:]:
        if '=' in token:
            key, value = token.split('=', 1)
            if key not in ('layout', 'name', 'azimuth', 'elevation', 'distance') or key in options:
                raise ValueError(f'Unknown or repeated video option: {key}')
            options[key] = value
        else:
            for view in token.split(','):
                view = ALIASES.get(view, view)
                if view not in VIEWS:
                    raise ValueError(f'Unknown view {view!r}; choose {", ".join(VIEWS)}')
                views.append(view)
    if not views:
        views = ['overview' if tokens[0] == 'arms' else 'front']
    if len(views) > 4 or len(set(views)) != len(views):
        raise ValueError('Choose 1–4 different views per video')
    layout = options.get('layout', layout)
    if layout not in ('grid', 'horizontal', 'vertical'):
        raise ValueError('layout must be grid, horizontal, or vertical')
    name = options.get('name', f'video{number}_{tokens[0]}_'+'_'.join(views))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', name):
        raise ValueError('Video name must use 1–100 letters, digits, hyphens or underscores')
    camera = {}
    for key in ('azimuth', 'elevation', 'distance'):
        if key in options:
            camera[key] = float(options[key])
            if not math.isfinite(camera[key]):
                raise ValueError('Camera values must be finite')
    if camera and len(views) != 1:
        raise ValueError('Custom camera values require a single view in that video')
    if camera.get('distance', 1) <= 0 or not -90 <= camera.get('elevation', 0) <= 90:
        raise ValueError('Camera distance must be positive; elevation must be -90..90 degrees')
    return dict(number=number, arms=tokens[0] in ('arms', 'native'), native_arms=tokens[0] == 'native', views=views, layout=layout, name=name, camera=camera)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    numbers = [int(m.group(1)) for value in argv if (m := re.fullmatch(r'--video(\d+)(?:=.*)?', value))]
    p = argparse.ArgumentParser(description='导出指定PT：一次物理回放，多视频/多视角，自动提交Slurm。',
        epilog='示例：--pt RUN/best.pt --videonums 2 --video1 arms backend --video2 noarms forhead\n'
               '拼接：--video1 arms front top layout=horizontal；视角：'+', '.join(VIEWS),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pt', '--checkpoint', dest='pt', type=Path, required=True)
    p.add_argument('--config', type=Path, help='默认PT旁的config.json')
    p.add_argument('--source-archive', type=Path, help='省略则按config源码SHA自动匹配项目内归档')
    p.add_argument('--recording', type=Path, help='复用同一PT的录制目录或已完成导出目录，免重跑策略')
    p.add_argument('--output', type=Path, help='必须为新目录；默认preview/outputs/export_北京时间')
    p.add_argument('--gpus', type=int, choices=(1, 2), default=2,
                   help='Slurm申请GPU数量；完整回放/渲染支持单卡，默认保留2卡')
    p.add_argument('--videonums', type=int, help='可省略，由--videoN数量推导；支持1–16')
    for number in sorted(set(numbers) | {1}):
        p.add_argument(f'--video{number}', nargs='+', metavar='SPEC', help='arms/noarms/native 视角... [layout=grid] [name=名称]')
    p.add_argument('--layout', choices=('grid', 'horizontal', 'vertical'), default='grid')
    p.add_argument('--plots', nargs='+', choices=(*PLOTS, 'all', 'none'),
                   default=['arm-joints', 'hand-joints'],
                   help='默认arm-joints hand-joints：每个臂/手关节的速度和加速度图；none关闭，all另含TCP速度')
    p.add_argument('--fps', type=finite, help='输出帧率，默认检查点控制频率')
    p.add_argument('--size', default='1280x720', help='每个视角的像素尺寸，偶数宽x高')
    p.add_argument('--piano-fill', type=finite, default=1.,
                   help='钢琴占每个画面宽度的比例，默认1铺满；正交投影，无透视变形')
    p.add_argument('--start', type=finite, default=0., help='截取起点秒，含准备阶段，默认0')
    p.add_argument('--end', type=finite, help='截取终点秒，默认完整轨迹')
    p.add_argument('--speed', type=finite, default=1., help='播放倍速，默认1；曲线仍用原时间')
    p.add_argument('--hold', type=finite, default=1., help='视频末帧停留秒数，默认1')
    p.add_argument('--audio', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--overlay', action=argparse.BooleanOptionalAction, default=True, help='视角/时间/检查点文字')
    p.add_argument('--key-highlight', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fingering', action=argparse.BooleanOptionalAction, default=True,
                   help='显示原乐谱目标琴键颜色和底部指法说明，不绘制悬浮标签，默认开启')
    p.add_argument('--zip', action='store_true', help='另外打包视频/图表/CSV/报告，不复制PT和模型')
    p.add_argument('--arm-config', type=Path, help='覆盖当前M6安装配置')
    p.add_argument('--connector-config', type=Path, help='覆盖当前手腕连接件配置')
    p.add_argument('--arm-ik', choices=('constrained', 'raw'), default='constrained',
                   help='默认constrained：原节奏整段优化并限制速度/加速度；raw为旧差分IK诊断')
    p.add_argument('--arm-position-tolerance-mm', type=finite, default=.1,
                   help='机械臂手腕位置误差验收上限，默认0.1 mm；不满足则导出失败')
    p.add_argument('--arm-rotation-tolerance-deg', type=finite, default=.5,
                   help='机械臂手腕姿态误差验收上限，默认0.5度；不满足则导出失败')
    p.add_argument('--dry-run', action='store_true', help='只校验并打印计划，不写文件、不提交作业')
    a = p.parse_args(argv)
    try:
        if len(numbers) != len(set(numbers)):
            raise ValueError('Each --videoN may occur only once')
        if not numbers:
            numbers = [1]
            a.video1 = ['noarms', 'front']
        count = a.videonums if a.videonums is not None else max(numbers)
        if not 1 <= count <= 16 or sorted(numbers) != list(range(1, count+1)):
            raise ValueError('--videonums and --video1..N must be consecutive and agree (1–16)')
        a.videos = [video_spec(getattr(a, f'video{n}'), n, a.layout) for n in numbers]
        a.videos.sort(key=lambda v: v['number'])
        if len({v['name'] for v in a.videos}) != count:
            raise ValueError('Video names must be unique')
        size = re.fullmatch(r'(\d+)x(\d+)', a.size)
        if not size:
            raise ValueError('--size must be WIDTHxHEIGHT')
        a.width, a.height = map(int, size.groups())
        if any(n % 2 or not 240 <= n <= 4096 for n in (a.width, a.height)):
            raise ValueError('Each size must be even and between 240 and 4096')
        if a.start < 0 or a.end is not None and a.end <= a.start:
            raise ValueError('Require 0 <= start < end')
        if a.fps is not None and not 1 <= a.fps <= 240 or not .05 <= a.speed <= 20 or not 0 <= a.hold <= 60:
            raise ValueError('fps: 1–240; speed: .05–20; hold: 0–60')
        if not .1 <= a.piano_fill <= 1:
            raise ValueError('--piano-fill must be between .1 and 1')
        if len(a.plots) > 1 and ('none' in a.plots or 'all' in a.plots):
            raise ValueError('Use all/none alone, or list individual plot groups')
        a.plots = list(PLOTS) if a.plots == ['all'] else [] if a.plots == ['none'] else sorted(set(a.plots))
        if a.arm_position_tolerance_mm <= 0 or not 0 < a.arm_rotation_tolerance_deg <= 180:
            raise ValueError('Arm position tolerance must be positive; rotation tolerance must be in (0, 180] degrees')
        if a.arm_ik == 'constrained' and a.speed != 1 and any(v['arms'] for v in a.videos):
            raise ValueError('Constrained arm videos preserve original tempo: use --speed 1; raw is diagnostic only')
    except ValueError as error:
        p.error(str(error))
    return a


def contract_file(name):
    return (name.startswith('tsensei_v3_2_pianist/') and '__pycache__' not in name.split('/')
            and not name.endswith(('.pyc', '.pyo'))) or bool(re.fullmatch(r'configs/actions/[^/]+\.json', name))


def archive_matches(path, expected):
    """Check bytes, not a basename or run date; final loader rechecks the contract."""
    manifest_path = Path(str(path)+'.manifest.json')
    if manifest_path.is_file():
        m = json.loads(manifest_path.read_text())
        actual = {k: v['sha256'] for k, v in m.get('files', {}).items() if contract_file(k)}
        if actual != expected:
            return False
        if m.get('archive_sha256') == sha256(path):
            return True
    actual = {}
    with tarfile.open(path, 'r:*') as archive:
        for member in archive.getmembers():
            if member.isfile() and contract_file(member.name):
                actual[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return actual == expected


def select_archive(config, explicit=None, root=ROOT):
    expected = config.get('source_runtime', {}).get('source_sha256')
    if not expected:
        raise ValueError('Only source_config checkpoints with complete source fingerprints are supported')
    candidates = [explicit] if explicit else sorted((root/'logs/hpc').glob('*/source.tar.gz'), reverse=True)
    if not explicit:
        candidates += sorted((root/'training_mjlab/source_versions').glob('*.tar.gz'), reverse=True)
    for path in candidates:
        if path.is_file() and archive_matches(path, expected):
            return path.resolve()
    raise ValueError('No matching source archive. Supply --source-archive with the checkpoint’s exact training source')


def resolve_recording(path):
    """Accept a raw recording or a clean export that points to its job cache."""
    path = path.resolve()
    if (path/'physics_trace.npz').is_file():
        return path
    report_path = path/'report.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get('recording'):
            cached = Path(report['recording'])
            return (cached if cached.is_absolute() else path/cached).resolve()
    return path


def build_plan(a):
    pt = a.pt.resolve()
    config_path = (a.config or pt.parent/'config.json').resolve()
    if not pt.is_file() or not config_path.is_file():
        raise ValueError('PT and complete config.json must exist')
    config = json.loads(config_path.read_text())
    native = bool(config.get('arm_follow'))
    if any(v.get('native_arms', False) != native for v in a.videos):
        raise ValueError('A trained arm checkpoint requires native videos; hand-only checkpoints cannot use native')
    if native and 'arm-joints' in a.plots:
        raise ValueError('Native arm export currently supports hand-joints/tcp plots; no reconstructed IK overlay')
    checkpoint_sha = sha256(pt)
    recording = resolve_recording(a.recording) if a.recording else None
    if recording:
        for name in ('report.json', 'physics_trace.npz', 'kinematic_scene/scene.xml'):
            if not (recording/name).is_file():
                raise ValueError('Recording is missing '+name)
        report = json.loads((recording/'report.json').read_text())
        if report['checkpoint_sha256'] != checkpoint_sha or report['source_runtime'] != config['source_runtime']:
            raise ValueError('Recording does not match this PT and config')
        if a.audio and not (recording/'played.wav').is_file():
            raise ValueError('Recording has no audio; use --no-audio or make a new recording')
    archive = None
    if not recording or a.fingering:
        archive = select_archive(config, a.source_archive.resolve() if a.source_archive else None)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = (a.output or ROOT/f'preview/outputs/export_{pt.stem}_{stamp}').resolve()
    if output.exists():
        raise ValueError('Output already exists; select a new directory (existing exports are never overwritten)')
    plan = dict(schema=1, output=str(output), checkpoint=str(pt), checkpoint_sha256=checkpoint_sha,
                checkpoint_origin=str(pt), config=str(config_path), config_sha256=sha256(config_path),
                source_archive=str(archive) if archive else None, archive_sha256=sha256(archive) if archive else None,
                recording=str(recording) if recording else None, videos=a.videos, plots=a.plots,
                fps=a.fps, width=a.width, height=a.height, piano_fill=a.piano_fill,
                start=a.start, end=a.end, speed=a.speed, hold=a.hold,
                audio=a.audio, overlay=a.overlay, key_highlight=a.key_highlight,
                fingering=a.fingering,  zip=a.zip, gpus=a.gpus,
                source_project=str(ROOT), arm_config=None, connector_config=None)
    plan.update(arm_ik=a.arm_ik, arm_position_tolerance_mm=a.arm_position_tolerance_mm,
                arm_rotation_tolerance_deg=a.arm_rotation_tolerance_deg)
    if recording:
        names = ['report.json', 'physics_trace.npz']
        names += [str(p.relative_to(recording)) for p in sorted((recording/'kinematic_scene').rglob('*')) if p.is_file()]
        names += [n for n in ('played.wav', 'preparation.npz', 'rollout.npz', 'physics_trace.json',
                              'audio_substeps.npz', 'audio_events.json', 'played.mid', 'ending.npz', 'ending.json') if (recording/n).is_file()]
        if a.fingering and not (recording/'rollout.npz').is_file():
            raise ValueError('Fingering requires the recording’s original rollout.npz')
        plan['recording_sha256'] = {n: sha256(recording/n) for n in names}
    for name in ('arm_config', 'connector_config'):
        path = getattr(a, name)
        if path:
            if not path.is_file():
                raise ValueError('Missing '+str(path))
            json.loads(path.read_text())
            plan[name] = str(path.resolve())
    return plan


def main():
    if '--trajectory-bundle' in sys.argv[1:]:
        from export_trajectory_bundle import main as trajectory_main
        return trajectory_main()
    a = parse_args()
    try:
        plan = build_plan(a)
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        raise SystemExit(str(error))
    if a.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    output = Path(plan['output'])
    inputs = output/'inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    # Freeze best.pt now, before it can be replaced by another training evaluation.
    for key, name in (('checkpoint', 'checkpoint.pt'), ('config', 'config.json'),
                      ('arm_config', 'arm_config.json'), ('connector_config', 'connector_config.json')):
        if plan[key]:
            source = Path(plan[key]); target = inputs/name
            shutil.copy2(source, target)
            expected = plan.get(key+'_sha256')
            if expected and sha256(target) != expected:
                raise SystemExit('Input changed while copying; no job submitted. Choose a fresh output and retry.')
            plan[key] = str(target)
    plan_path = output/'export_plan.json'
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    result = subprocess.run(['bash', str(ROOT/'scripts/hpc/submit.sh'), '--gpus', str(a.gpus),
                             'export', '--plan', str(plan_path)],
                            text=True, capture_output=True)
    if result.returncode:
        raise SystemExit('Slurm submission failed; plan preserved at '+str(plan_path)+'\n'+result.stderr)
    job = result.stdout.strip().split(';')[0]
    (output/'submission.json').write_text(json.dumps({'job_id':job, 'plan':str(plan_path)}, indent=2)+'\n')
    print(f'Slurm job: {job}\nOutput: {output}\nLog: {ROOT}/logs/hpc/{job}.out\n'
          f'Status: {output}/status.json\nAfter completion: {output}/index.html')


if __name__ == '__main__':
    main()
