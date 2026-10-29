"""Portable numerical reproduction from recorded joints and the captured MDH model.

NumPy only. Reads local JSON files and prints results without writing configuration.
The model is a recorded nominal model, not an independent physical measurement.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from example import load_calibration, transform_points, flange_pose_from_contact


def fk_mdh(q_deg, rows):
    result = np.eye(4)
    for i, (alpha, a, d, theta) in enumerate(rows):
        a_rad = np.deg2rad(alpha)
        t_rad = np.deg2rad(theta+(q_deg[i] if i < 7 else 0.))
        ca, sa, ct, st = np.cos(a_rad), np.sin(a_rad), np.cos(t_rad), np.sin(t_rad)
        rx = np.array([[1., 0, 0], [0, ca, -sa], [0, sa, ca]])
        rz = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1.]])
        step = np.eye(4)
        step[:3, :3] = rx @ rz
        step[:3, 3] = np.array([a, 0., 0.])+rx @ rz @ np.array([0., 0., d])
        result = result @ step
    return result


def fit(source, target):
    u, _, vt = np.linalg.svd((source-source.mean(0)).T @ (target-target.mean(0)))
    r = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, target.mean(0)-r @ source.mean(0)
    return t


def check(actual, expected, atol=1e-8):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def main():
    cal = load_calibration()
    path = Path(__file__).with_name(cal["inputs_file"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == cal["inputs_sha256"]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows, points = data["right_arm_mdh_rows"], data["points"]
    keep = np.array([p["used_for_fit"] for p in points])
    assert [p["id"] for p in points if p["used_for_fit"]] == cal["fitted_point_ids"]
    assert keep.sum() == 9 and not keep[0] and points[0]["id"] == "p1"
    assert points[0]["arm_deg"] == [-48.64, -70.97, 34.89, -90.87, 131.72, -45.82, -54.49]
    P = np.array([p["piano_mm"] for p in points])
    F = np.array([fk_mdh(p["arm_deg"], rows) for p in points])
    check(F, [p["T_B_from_F_mm"] for p in points])
    tip = np.array(cal["probe_tip_in_F_mm"])
    check(tip, [0., 0., 156.])
    B = F[:, :3, 3] + np.einsum("nij,j->ni", F[:, :3, :3], tip)
    check(B, [p["model_tip_B_mm"] for p in points])
    fitted = fit(P[keep], B[keep])
    T_BP, T_PB = np.array(cal["T_B_from_P_mm"]), np.array(cal["T_P_from_B_mm"])
    check(fitted, T_BP)
    check(T_BP @ T_PB, np.eye(4), 1e-12)
    check(T_BP[:3, :3].T @ T_BP[:3, :3], np.eye(3), 1e-12)
    check(np.linalg.det(T_BP[:3, :3]), 1., 1e-12)
    error = transform_points(T_PB, B)-P
    check(error, [p["residual_P_mm"] for p in points])
    norms = np.linalg.norm(error, axis=1)
    quality = cal["quality"]
    check(np.sqrt(np.mean(norms[keep]**2)), quality["rms_3d_mm"])
    check(norms[keep].max(), quality["max_3d_mm"])
    check(norms[0], cal["excluded_validation"]["error_3d_mm"])
    check(np.sqrt(np.mean(error[keep]**2, axis=0)), quality["axis_rms_P_mm"])
    check(np.linalg.norm(transform_points(T_BP, P)-B, axis=1), norms)
    check(transform_points(T_PB, transform_points(T_BP, P)), P)
    for stem in ["T_B_from_P", "T_P_from_B"]:
        mm, m = np.array(cal[stem+"_mm"]), np.array(cal[stem+"_m"])
        check(m[:3, :3], mm[:3, :3], 1e-12)
        check(m[:3, 3], mm[:3, 3]/1000, 1e-12)
        check(m[3], mm[3], 1e-12)
    check(transform_points(cal["T_B_from_P_m"], P/1000), transform_points(T_BP, P)/1000, 1e-12)

    loo = []
    selected = np.flatnonzero(keep)
    for i in selected:
        training = selected[selected != i]
        sub = fit(P[training], B[training])
        loo.append(np.linalg.norm(transform_points(sub, P[i])-B[i]))
    check(np.sqrt(np.mean(np.square(loo))), quality["leave_one_out_rms_mm"])
    check(max(loo), quality["leave_one_out_max_mm"])
    for i in selected:
        pose = flange_pose_from_contact(T_BP, P[i], F[i, :3, :3], tip)
        check(pose[:3, 3]+pose[:3, :3] @ tip, transform_points(T_BP, P[i]))
    print("Recorded MDH FK, proper nine-point fit, inverse, units and probe compensation verified.")
    print(f"Selected nine-point RMS: {quality['rms_3d_mm']:.6f} mm")
    print(f"Selected max residual: {quality['max_3d_mm']:.6f} mm ({quality['max_point']})")
    print(f"Nine-point leave-one-out RMS: {quality['leave_one_out_rms_mm']:.6f} mm")
    print(f"Corrected p1 held-out error: {norms[0]:.6f} mm")
    print("No hardware connection or configuration write is performed.")


if __name__ == "__main__":
    main()
