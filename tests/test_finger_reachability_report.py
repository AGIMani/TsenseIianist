"""Emit a per-finger reachability report for the current hand contract."""
import json
from pathlib import Path

from tests.test_finger_coupling import CouplingGeometryTests, FINGER_NAMES, RANGE_MODE


def test_current_finger_reachability_report():
    # Reuse the exact compiled scene and reference-pose solver used by the
    # coupling regression, then expose one auditable line per finger.
    CouplingGeometryTests.setUpClass()
    options = {
        'schema': 1,
        'mode': RANGE_MODE,
        'angle_ranges_deg': json.loads(
            (Path(__file__).resolve().parents[1] / 'configs/finger_coupling/angle_ranges_60_80.json').read_text()),
    }
    coupler = CouplingGeometryTests().range_coupler(options)
    diagnostics = coupler.reference_diagnostics
    angles = diagnostics['coupling_target_angle_deg']
    errors = diagnostics['coupling_target_error_deg']
    infeasible = diagnostics['coupling_infeasible']
    at_limit = diagnostics['coupling_follower_at_limit']
    for name, angle, error, bad, limited in zip(FINGER_NAMES, angles, errors, infeasible, at_limit):
        status = 'REACHABLE' if not bad else 'UNREACHABLE'
        print(f'FINGER_REACH {name} {status} target_deg={angle:.6f} error_deg={error:.6f} follower_at_limit={bool(limited)}', flush=True)
    assert not any(infeasible)


def test_current_yaw30_active_key_reachability():
    """Run the real-geometry endpoint audit for every finger under current limits."""
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts' /
                      'probe_active_pad_reachability.py'), run_name='__main__')
    report = (Path(__file__).resolve().parents[1] /
              'preview/outputs/diagnostics/finger_clearance_20260927/active_pad_reachability.json')
    data = json.loads(report.read_text())
    assert len(data['cases']) == 12
    assert all(case['best']['pad_allowed'] for case in data['cases'])

