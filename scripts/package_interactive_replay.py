"""Package same-rollout physics and verified score annotations for portable MuJoCo replay."""
import json, shutil, subprocess, sys, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def install_local_launchers(folder):
    text=(ROOT/'scripts/launch_local_recorded_replay.sh').read_text()
    for name in ('launch_local.sh','launch.sh','launch_cpu.sh','launch_remote.sh'):
        path=folder/name;path.write_text(text);path.chmod(0o755)
    (folder/'README.txt').write_text(
        '本地 MuJoCo 录制回放：运行 ./launch_local.sh（或 ./launch.sh）。\n'
        '旧名称 launch_remote.sh / launch_cpu.sh 现在也启动本地回放，不再连接 SSH。\n'
        '读取同次录制的 model.mjb，不编译 XML、不运行策略或重新仿真。\n'
        '默认 NVIDIA PRIME/RTX OpenGL 渲染，本地 ffplay 播放真实触键声音。\n'
        '自动查找项目 training_mjlab/.venv；独立复制时设置 PIANIST_REPLAY_PYTHON，MuJoCo版本须与replay.json匹配。\n'
        '需至少4GiB可用系统内存；独立systemd用户scope限制内存3GiB、禁止swap，不能绕过失败保护。\n'
        '单线程GLFW窗口；上传网格完成后显示窗口；默认关闭阴影与反射，保留原模型几何。\n'
        '无手指或琴键悬浮标签；保留琴键颜色提示及实际触键高亮。\n'
        'R重置时间和声音并立即从头重播（保持当前速度）；Space播放/暂停，左右方向键跳转1秒，底部进度条可点击/拖动，1～4切换总览/近景/侧面/顶视，A切换实际手根/法兰坐标轴，Home回到开头，-/=调整速度，鼠标旋转/平移/缩放。\n'
        './launch_local.sh --validate 检查数据；--render-check 检查本地NVIDIA渲染。\n')

def install_replay_display(folder):
    # Never copy the frozen training helper: old experiments can contain labels.
    shutil.copy2(ROOT/'scripts/interactive_recorded_replay.py',folder/'viewer.py')
    shutil.copy2(ROOT/'scripts/replay_key_highlight.py',folder/'preview_fingering.py')


def package(output,recording,plan):
    import numpy as np
    from tsensei_v3_2_pianist.key_activation import validate_fraction
    with np.load(recording/'audio_substeps.npz', allow_pickle=False) as samples:
        if 'key_threshold' in samples or 'key_activation_fraction' not in samples:
            raise ValueError('Old activation recording requires its original frozen replay tools')
        fraction=validate_fraction(float(samples['key_activation_fraction']))
    folder=output/'interactive_replay';folder.mkdir(exist_ok=True)
    report=json.loads((recording/'report.json').read_text());annotations=json.loads((output/'data/score_annotations.json').read_text())
    if report['checkpoint_sha256']!=plan['checkpoint_sha256'] or annotations['checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('Interactive provenance differs')
    for n in ('physics_trace.npz','played.wav','report.json','rollout.npz'):
        shutil.copy2(recording/n,folder/n)
    if (recording/'preparation.npz').is_file():shutil.copy2(recording/'preparation.npz',folder/'preparation.npz')
    shutil.copytree(recording/'kinematic_scene',folder/'kinematic_scene',dirs_exist_ok=True)
    for n in ('score_annotations.npz','score_annotations.json'):shutil.copy2(output/'data'/n,folder/n)
    install_replay_display(folder)
    for n in ('preview_keys.py','key_activation.py'):shutil.copy2(ROOT/'tsensei_v3_2_pianist'/n,folder/n)
    shutil.copy2(recording/'kinematic_scene/model.mjb',folder/'model.mjb')
    (folder/'replay.py').write_text('from viewer import main\nif __name__ == \"__main__\": main()\n')
    meta=dict(display_contract='piano_key_colors_only_v1',key_activation_fraction=fraction,checkpoint_sha256=plan['checkpoint_sha256'],archive_sha256=plan['archive_sha256'],native_arms=bool(plan['videos'][0].get('native_arms',False)),dependency_versions=report['source_runtime']['versions']['learner'],preparation_prefix_seconds=report.get('preparation_prefix_seconds',0),fingertip_sites=annotations['fingertip_sites'],source_recording=str(recording),physics_trace_sha256=__import__('hashlib').sha256((recording/'physics_trace.npz').read_bytes()).hexdigest())
    (folder/'replay.json').write_text(json.dumps(meta,indent=2)+'\n')
    shutil.copy2(ROOT/'scripts/launch_remote_recorded_replay.py',folder/'launch_remote.py')
    (folder/'remote_launch.json').write_text(json.dumps(dict(host='volc-a32',python=str(ROOT/'training_mjlab/.venv/bin/python'),replay=str(folder/'replay.py')),indent=2)+'\n')
    install_local_launchers(folder)
    # Never open a window/render on the SSH/login host. This is a CPU data check.
    env=os.environ.copy();env['MUJOCO_GL']='disable'
    python=ROOT/'training_mjlab/.venv/bin/python'
    completed=subprocess.run([str(python),str(folder/'replay.py'),'--validate'],env=env,text=True,capture_output=True)
    (folder/'validation.log').write_text(completed.stdout+completed.stderr)
    if completed.returncode:raise RuntimeError('Interactive replay data validation failed: '+completed.stderr[-2000:])
    return json.loads(completed.stdout.splitlines()[-1])
