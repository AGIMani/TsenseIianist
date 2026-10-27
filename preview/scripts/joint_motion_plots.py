"""CPU plots of signed joint velocity and acceleration from recorded positions."""
import json
import math
from pathlib import Path
import re

import numpy as np


def derivatives(time, positions):
    time = np.asarray(time, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if (time.ndim != 1 or len(time) < 2 or positions.ndim != 2 or
            positions.shape[0] != len(time) or positions.shape[1] < 1 or
            not np.isfinite(time).all() or not np.isfinite(positions).all() or
            np.any(np.diff(time) <= 0)):
        raise ValueError('Require finite positions and at least two strictly increasing timestamps')
    velocity_time = (time[1:]+time[:-1])/2
    velocity = np.diff(positions, axis=0)/np.diff(time)[:, None]
    acceleration_time = (velocity_time[1:]+velocity_time[:-1])/2
    acceleration = np.diff(velocity, axis=0)/np.diff(velocity_time)[:, None]
    return velocity_time, velocity, acceleration_time, acceleration


def joint_slug(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_')


def write_csv(path, time, values, names):
    np.savetxt(path, np.column_stack((time, values)), delimiter=',',
               header=','.join(['original_time_s', *names]), comments='', fmt='%.17g')


def export_joint_motion(folder, time, positions, joints, label, *, limits=None, continuous_path=None,
                        position_source='bounded_trajectory'):
    """Each joint gets separate PNG/PDF plots; each side also gets two overviews.

    Recorded positions use finite differences. A supplied planned path uses its
    analytic derivatives, including knots where acceleration reaches extrema.
    No filtering, refitting, retiming, or final-video-hold samples are introduced.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    time = np.asarray(time, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if position_source not in ('bounded_trajectory', 'recorded_physics'):
        raise ValueError('Unknown position source')
    if position_source == 'recorded_physics' and continuous_path is not None:
        raise ValueError('Recorded physics cannot use a planned continuous path')
    vt, velocity, at, acceleration = derivatives(time, positions)
    if len(joints) != positions.shape[1]:
        raise ValueError('Joint metadata must match position columns')
    bounded = [j for j in joints if j.get('position_min') is not None and j.get('position_max') is not None]
    position_audit = []
    for i, joint in enumerate(joints):
        lo, hi = joint.get('position_min'), joint.get('position_max')
        if (lo is None) != (hi is None):
            raise ValueError('Each position limit requires both bounds')
        if lo is None:
            position_audit.append(None)
            continue
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('Position limits must be finite and increasing')
        values = positions[:, i]
        below, above = values < lo, values > hi
        exceeded = below | above
        times = time[exceeded]
        position_audit.append(dict(minimum=float(values.min()), maximum=float(values.max()),
            lower_limit=float(lo), upper_limit=float(hi), unit=joint['unit'],
            below_limit_samples=int(below.sum()), above_limit_samples=int(above.sum()),
            exceeded_samples=int(exceeded.sum()), exceeded_sample_fraction=float(exceeded.mean()),
            maximum_undershoot=float(np.maximum(lo-values, 0).max()),
            maximum_overshoot=float(np.maximum(values-hi, 0).max()),
            first_exceedance_s=float(times[0]) if len(times) else None,
            last_exceedance_s=float(times[-1]) if len(times) else None))
    if bounded:
        if position_source == 'bounded_trajectory':
            if len(bounded) != len(joints):
                raise ValueError('Position limits must cover every exported joint')
            from tsensei_v3_2_pianist.mjlab_env.joint_limits import require_bounded
            require_bounded(positions, [j['position_min'] for j in joints], [j['position_max'] for j in joints])
    if continuous_path is not None:
        np.testing.assert_allclose(positions, continuous_path(time), rtol=0, atol=1e-10,
                                   err_msg='Plotted positions must use the rendered continuous path')
        knots = continuous_path.t
        dense = np.unique(np.r_[time, (time[:-1]+time[1:])/2, knots[(knots >= time[0]) & (knots <= time[-1])]])
        vt = at = dense
        velocity, acceleration = continuous_path(dense, 1), continuous_path(dense, 2)
        if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
            raise ValueError('Nonfinite analytic path derivatives')
    names = [j['name'] for j in joints]
    slugs = [joint_slug(name) for name in names]
    if (len(joints) != positions.shape[1] or len(set(slugs)) != len(slugs) or
            any(not s for s in slugs) or
            any(j['unit'] not in ('rad', 'm') or j['side'] not in ('right', 'left') for j in joints)):
        raise ValueError('Joint names, sides and position units must match the recorded columns')
    if limits is not None:
        if set(limits['joints']) != set(names):
            raise ValueError('Reference limits must match the plotted joint names exactly')
        for joint in joints:
            for kind, suffix in (('velocity', '/s'), ('acceleration', '/s^2')):
                value = limits['joints'][joint['name']][kind]
                if (limits['units'][kind] != joint['unit'] + suffix or isinstance(value, bool) or
                        not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                    raise ValueError('Reference limits must be positive and use matching joint units')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if limits is not None:
        (folder/'joint_limits.json').write_text(json.dumps(limits, indent=2)+'\n')
    write_csv(folder/'joint_positions.csv', time, positions, [f'{s}_{j["unit"]}' for s, j in zip(slugs, joints)])
    np.testing.assert_array_equal(
        np.loadtxt(folder/'joint_positions.csv', delimiter=',', skiprows=1, ndmin=2)[:, 1:], positions,
        err_msg='Exported positions must preserve every input sample exactly')
    write_csv(folder/'joint_velocities.csv', vt, velocity, [f'{s}_{j["unit"]}_s' for s, j in zip(slugs, joints)])
    write_csv(folder/'joint_accelerations.csv', at, acceleration, [f'{s}_{j["unit"]}_s2' for s, j in zip(slugs, joints)])
    np.savez_compressed(folder/'joint_motion_samples.npz', time=time, positions=positions,
                        velocity_time=vt, velocity=velocity, acceleration_time=at, acceleration=acceleration,
                        joint_names=np.asarray(names), position_units=np.asarray([j['unit'] for j in joints]))
    summary = []
    for i, joint in enumerate(joints):
        item = dict(joint)
        item['position'] = position_audit[i]
        for kind, t, values in (('velocity', vt, velocity), ('acceleration', at, acceleration)):
            index = int(np.argmax(np.abs(values[:, i]))) if len(t) else None
            item[kind] = None if index is None else dict(
                maximum_absolute=float(abs(values[index, i])), signed_at_peak=float(values[index, i]),
                peak_time_s=float(t[index]), rms=float(np.sqrt(np.mean(values[:, i]**2))))
            if limits is not None and index is not None:
                limit = limits['joints'][joint['name']][kind]
                exceeded = np.abs(values[:, i]) > limit
                exceeded_times = t[exceeded]
                item[kind]['reference_limit'] = limit
                item[kind]['peak_to_limit_ratio'] = item[kind]['maximum_absolute']/limit
                item[kind]['exceeded_samples'] = int(exceeded.sum())
                item[kind]['exceeded_sample_fraction'] = float(exceeded.mean())
                item[kind]['first_exceedance_s'] = float(exceeded_times[0]) if len(exceeded_times) else None
                item[kind]['last_exceedance_s'] = float(exceeded_times[-1]) if len(exceeded_times) else None
        summary.append(item)

    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'path.simplify': False}):
        for i, audit in enumerate(position_audit):
            if audit is None:
                continue
            fig, ax = plt.subplots(figsize=(11, 3.5))
            values = positions[:, i]
            ax.plot(time, values, lw=.8, label='Recorded position' if position_source == 'recorded_physics' else 'Trajectory position')
            ax.axhline(audit['lower_limit'], color='#b32626', ls='--', label='Model position bounds')
            ax.axhline(audit['upper_limit'], color='#b32626', ls='--')
            exceeded = (values < audit['lower_limit']) | (values > audit['upper_limit'])
            if exceeded.any():
                ax.scatter(time[exceeded], values[exceeded], color='#b32626', s=8, zorder=3,
                           label=f'Outside bounds: {int(exceeded.sum())} samples')
            ax.set(xlabel='Original replay time (s)', ylabel=joints[i]['unit'],
                   title=f'{label} | {joints[i]["name"]} | position | unchanged samples',
                   xlim=(time[0], time[-1]))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
            fig.tight_layout()
            for ext in ('png', 'pdf'):
                fig.savefig(folder/f'{slugs[i]}_position.{ext}', dpi=150)
            plt.close(fig)
        for kind, plural, t, values, color in (
                ('velocity', 'velocities', vt, velocity, '#2377b5'),
                ('acceleration', 'accelerations', at, acceleration, '#ca6231')):
            suffix = '/s' if kind == 'velocity' else '/s²'

            def draw(ax, i, small=False):
                joint = joints[i]
                limit = limits['joints'][joint['name']][kind] if limits is not None else None
                if len(t):
                    ax.plot(t, values[:, i], color=color, lw=.65 if small else .8)
                    peak = summary[i][kind]['maximum_absolute']
                    peak_text = f'peak |{kind[0]}| = {peak:.4g}'
                    if limit is not None:
                        mask = np.abs(values[:, i]) > limit
                        if mask.any():
                            ax.scatter(t[mask], values[mask, i], color='#b32626', s=5,
                                       linewidths=0, zorder=3, label='Above SDK limit')
                        peak_text += f' ({peak/limit:.2f}x limit)'
                    ax.text(.99, .94, peak_text, transform=ax.transAxes,
                            ha='right', va='top', fontsize=8, color='#455361',
                            bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1))
                else:
                    ax.text(.5, .5, 'Insufficient samples', ha='center', va='center', transform=ax.transAxes)
                if limit is not None:
                    ax.axhline(limit, color='#b32626', ls='--', lw=.9,
                               label=f'SDK ±{limit:.4g} {joint["unit"]}{suffix}')
                    ax.axhline(-limit, color='#b32626', ls='--', lw=.9)
                    ax.legend(loc='lower left', fontsize=7 if small else 8, framealpha=.9)
                else:
                    ax.text(.01, .03, 'Reference limit unavailable', transform=ax.transAxes,
                            fontsize=7 if small else 8, color='#65717c',
                            bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1))
                ax.axhline(0, color='#74828e', lw=.5, alpha=.7)
                ax.set_xlim(time[0], time[-1])
                ax.grid(alpha=.2)
                ax.set_ylabel(joint['unit']+suffix)
                ax.set_title(joint['name'].split('/', 1)[-1], fontsize=9 if small else 12)
                ax.tick_params(labelsize=8 if small else 10)

            for i, slug in enumerate(slugs):
                fig, ax = plt.subplots(figsize=(11, 3.5))
                draw(ax, i)
                ax.set_title(f'{label} | {joints[i]["side"]} | {joints[i]["name"].split("/", 1)[-1]} | {kind}')
                ax.set_xlabel('Original replay time (s)')
                if limits is not None:
                    fig.text(.99, .005, 'M6 SDK configuration reference; controller limits not queried',
                             ha='right', fontsize=7, color='#65717c')
                fig.tight_layout(rect=(0, .025 if limits is not None else 0, 1, 1))
                for ext in ('png', 'pdf'):
                    fig.savefig(folder/f'{slug}_{kind}.{ext}', dpi=150)
                plt.close(fig)

            for side in ('right', 'left'):
                indices = [i for i, j in enumerate(joints) if j['side'] == side]
                if not indices:
                    continue
                columns = 2 if len(indices) <= 7 else 3
                rows = math.ceil(len(indices)/columns)
                fig, axes = plt.subplots(rows, columns, figsize=(13 if columns == 2 else 17, rows*2.5),
                                         sharex=True, squeeze=False)
                for ax in axes.flat:
                    ax.set_visible(False)
                for i, ax in zip(indices, axes.flat):
                    ax.set_visible(True)
                    draw(ax, i, small=True)
                    ax.set_xlabel('Time (s)', fontsize=8)
                    ax.tick_params(labelbottom=True)
                method = 'analytic spline' if continuous_path is not None else 'unfiltered differences'
                fig.suptitle(f'{side.capitalize()} {label} joint {plural} | original time | {method}', fontsize=14)
                if limits is not None:
                    fig.text(.99, .005, 'M6 SDK configuration reference; controller limits not queried',
                             ha='right', fontsize=8, color='#65717c')
                fig.tight_layout(rect=(0, .025 if limits is not None else 0, 1, .975))
                for ext in ('png', 'pdf'):
                    fig.savefig(folder/f'{side}_joint_{plural}.{ext}', dpi=150)
                plt.close(fig)

    report = dict(schema=2, label=label, joints=summary, position_samples=len(time),
                  position_source=position_source, positions_modified=False,
                  position_bounds_status=('exceeded' if any(a and a['exceeded_samples'] for a in position_audit)
                                          else 'within_bounds' if len(bounded) == len(joints) else 'partially_unavailable'),
                  position_limit_comparison='Strict measured q < lower or q > upper; no tolerance, clipping, projection or dropped samples',
                  velocity_samples=len(vt), acceleration_samples=len(at),
                  start_s=float(time[0]), end_s=float(time[-1]),
                  sampling_interval_s=dict(minimum=float(np.diff(time).min()), maximum=float(np.diff(time).max())),
                  time_basis='original replay time; video speed and final display hold excluded',
                  velocity_definition='v[i] = (q[i+1]-q[i])/(t[i+1]-t[i]); timestamp = interval midpoint',
                  acceleration_definition='a[i] = (v[i+1]-v[i])/(tv[i+1]-tv[i]); timestamp = velocity-midpoint midpoint',
                  filtering='none; finite differences of recorded samples, no spline or endpoint padding',
                  reference_limits=limits,
                  limit_comparison='abs(sample) > positive reference; both signs plotted; no clipping or retiming',
                  hardware_validation='not performed; reference limits are not a dynamic feasibility guarantee')
    if continuous_path is not None:
        report.update(velocity_definition='Analytic first derivative of the rendered cubic B-spline',
                      acceleration_definition='Analytic second derivative of the rendered cubic B-spline',
                      filtering='none; samples, midpoints and spline knots; no refitting',
                      position_source='planned continuous trajectory, not the raw differential-IK seed')
    (folder/'joint_motion_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
