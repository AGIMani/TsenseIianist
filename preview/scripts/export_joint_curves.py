#!/usr/bin/env python3
"""Export joint velocity/acceleration plots for an already exported video, on CPU."""
import argparse
import html
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from export_videos import sha256
from export_video_job import crop_times, hand_motion_columns, interpolate, load_trace
from joint_motion_plots import export_joint_motion
from arm_motion_limits import load_m6_joint_limits, limits_table_html


def export(video, output=None, groups=('arm-joints', 'hand-joints')):
    import mujoco
    import numpy as np

    video = Path(video).resolve()
    source = video.parent.parent
    manifest = json.loads((source/'manifest.json').read_text())
    verified = {}

    def verify(name):
        path = source/name
        digest = sha256(path)
        if manifest.get(name) != digest:
            raise ValueError('Export input checksum mismatch: '+name)
        verified[str(path)] = digest
        return path

    report = json.loads(verify('report.json').read_text())
    video_name = str(video.relative_to(source))
    verify(video_name)
    matches = [v for v in report['videos'] if v['file'] == video_name]
    if len(matches) != 1 or matches[0]['sha256'] != verified[str(video)]:
        raise ValueError('Video does not match its export report')
    recording = Path(report['recording'])
    if not recording.is_absolute():
        recording = (source/recording).resolve()
    trace_path = recording/'physics_trace.npz'
    verified[str(trace_path)] = sha256(trace_path)
    if verified[str(trace_path)] != report['physics_trace_sha256']:
        raise ValueError('Original physics trace checksum mismatch')
    trace, recorded = load_trace(recording)
    if recorded['checkpoint_sha256'] != report['checkpoint_sha256']:
        raise ValueError('Recording and video checkpoint hashes differ')
    start, end = float(report['clip_start']), float(report['clip_end'])
    if not 0 <= trace['time'][0] <= start < end <= trace['time'][-1]+1e-8:
        raise ValueError('Video interval is outside the original recording')
    time = crop_times(trace['time'], start, min(end, trace['time'][-1]))
    sample = lambda values: np.stack([interpolate(values, trace['time'], t) for t in time])
    payloads = {}
    arm_path = None
    if 'arm-joints' in groups:
        with np.load(verify('data/arm_trajectory.npz'), allow_pickle=False) as arm:
            arm_time, q = arm['time'].copy(), arm['q'].copy()
        if not np.array_equal(arm_time, trace['time']) or q.shape != (len(arm_time), 14):
            raise ValueError('Arm and hand trajectories are not synchronized M6 data')
        names = [f'{side}_J{j}' for side in ('right', 'left') for j in range(1, 8)]
        angle_csv = verify('data/arm_joint_angles.csv')
        if angle_csv.open().readline().strip().split(',') != ['time_s', *[n+'_rad' for n in names]]:
            raise ValueError('Unrecognized arm joint column order')
        np.testing.assert_allclose(np.loadtxt(angle_csv, delimiter=',', skiprows=1),
                                   np.column_stack((arm_time, q)), rtol=1e-10, atol=1e-10)
        joints = [dict(name=n, side=n.split('_')[0], unit='rad') for n in names]
        if (report.get('arm_analysis') or {}).get('motion_plan') is not None:
            from constrained_arm_motion import load_path
            arm_path = load_path(verify('data/arm_spline.npz'))
            np.testing.assert_allclose(arm_path(arm_time), q, rtol=0, atol=1e-10)
        payloads['arm_joints'] = (arm_path(time) if arm_path is not None else sample(q), joints, 'M6 arm')
    if 'hand-joints' in groups:
        scene = recording/'kinematic_scene/scene.xml'
        verified[str(scene)] = sha256(scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        if model.nq != trace['qpos'].shape[1] or model.nq != model.nv:
            raise ValueError('Recorded scalar joint coordinates do not match the compiled scene')
        positions, joints = hand_motion_columns(model, sample(trace['qpos']))
        payloads['hand_joints'] = (positions, joints, 'hand / wrist')
    output = Path(output).resolve() if output else source/(video.stem+'_joint_curves')
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for name, (positions, joints, label) in payloads.items():
        print(f'Plotting {name}: {len(joints)} joints, velocity and acceleration', flush=True)
        summaries[name] = export_joint_motion(output/name, time, positions, joints, label,
                                              limits=load_m6_joint_limits() if name == 'arm_joints' else None,
                                              continuous_path=arm_path if name == 'arm_joints' else None)
    result = dict(schema=1, video=str(video), video_sha256=verified[str(video)],
                  source_export=str(source), source_job=report['source_job'], verified_inputs=verified,
                  clip_start_s=start, clip_end_s=end, video_speed=report['speed'],
                  video_duration_s=matches[0]['duration'], original_sample_hz=report['physics_hz'],
                  time_basis='original physics time, not video frame differences; final display hold excluded',
                  arm_motion_plan=(report.get('arm_analysis') or {}).get('motion_plan'),
                  limitations=['Arm curves use the saved video trajectory, not hardware measurements.',
                               'Planned splines use analytic derivatives; other curves use unfiltered finite differences.'],
                  groups=summaries)
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    blocks = ['<!doctype html><meta charset="utf-8"><title>关节速度与加速度</title>',
              '<style>body{font:16px sans-serif;max-width:1400px;margin:32px auto;padding:0 16px;color:#203040}'
              'img{max-width:100%}a{color:#186da8}li{margin:8px 0}</style>',
              '<h1>关节速度与加速度</h1><p>'+html.escape(video.name)+'</p>',
              f'<p>原始物理时间 {start:g}–{end:g} 秒，{report["physics_hz"]:g} Hz；不滤波，不加入视频末帧停留。</p>',
              '<p>旋转关节：rad/s、rad/s²；平移关节：m/s、m/s²。机械臂采用对应视频保存的轨迹。</p>',
              '<p><a href="joint_curves.zip">全部图表与数据ZIP</a> · <a href="report.json">数据来源与计算说明</a></p>']
    for name in summaries:
        blocks.append('<h2>'+('机械臂' if name == 'arm_joints' else '双手与腕部')+'</h2>')
        if summaries[name]['reference_limits'] is not None:
            blocks.append(limits_table_html(summaries[name]['reference_limits'], name))
        else:
            blocks.append('<p>手与腕部尚无已确认的限值，保留实际曲线并注明限值未提供。</p>')
        blocks.append(f'<p><a href="{name}/joint_velocities.csv">速度CSV</a> · '
                      f'<a href="{name}/joint_accelerations.csv">加速度CSV</a> · '
                      f'<a href="{name}/joint_motion_samples.npz">完整数值NPZ</a></p>')
        for side, side_label in (('right', '右侧'), ('left', '左侧')):
            for kind, label in (('velocities', '速度'), ('accelerations', '加速度')):
                stem = f'{name}/{side}_joint_{kind}'
                blocks.append(f'<h3>{side_label}{label}</h3><a href="{stem}.pdf">PDF</a>'
                              f'<a href="{stem}.png"><img loading="lazy" src="{stem}.png"></a>')
        blocks.append('<details><summary>每个关节单独的曲线</summary><ul>')
        for file in sorted((output/name).glob('*.png')):
            if '_joint_velocities' in file.stem or '_joint_accelerations' in file.stem:
                continue
            rel = html.escape(str(file.relative_to(output)))
            blocks.append(f'<li><a href="{rel}">{html.escape(file.stem)}</a> · <a href="{rel[:-4]}.pdf">PDF</a></li>')
        blocks.append('</ul></details>')
    (output/'index.html').write_text('\n'.join(blocks)+'\n')
    files = [p for p in output.rglob('*') if p.is_file()]
    (output/'manifest.json').write_text(json.dumps({str(p.relative_to(output)): sha256(p) for p in files}, indent=2)+'\n')
    with zipfile.ZipFile(output/'joint_curves.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for file in files+[output/'manifest.json']:
            bundle.write(file, str(file.relative_to(output)))
    print('Exported: '+str(output/'index.html'), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New directory; defaults beside the original videos/data folders')
    parser.add_argument('--plots', nargs='+', choices=('arm-joints', 'hand-joints'), default=['arm-joints', 'hand-joints'])
    args = parser.parse_args()
    export(args.video, args.output, args.plots)


if __name__ == '__main__':
    main()
