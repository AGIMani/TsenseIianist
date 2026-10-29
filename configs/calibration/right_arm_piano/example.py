"""Selected nine-point calibration: offline position and flange-pose examples.

Requires NumPy. Run `python example.py` from any working directory.
All helper inputs must use the same length unit as the selected transform.
"""
import json
from pathlib import Path
import numpy as np


def load_calibration(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name("transform.json")
    return json.loads(path.read_text(encoding="utf-8"))


def transform_points(transform, points):
    """Transform one (3,) point or (...,3) row-wise points to the destination frame."""
    transform = np.asarray(transform, dtype=float)
    return np.asarray(points, dtype=float) @ transform[:3, :3].T + transform[:3, 3]


def transform_vectors(transform, vectors):
    """Rotate displacement/linear-velocity vectors; no origin translation."""
    return np.asarray(vectors, dtype=float) @ np.asarray(transform, dtype=float)[:3, :3].T


def flange_pose_from_contact(T_B_from_P, contact_P, R_B_from_F_des, tip_in_F):
    """Return flange pose in base for a contact target and a chosen flange orientation.

    tip_in_F includes tool geometry. For the probe use [0,0,156] in mm;
    for a hand supply the tip vector computed at the intended hand posture.
    """
    rotation = np.asarray(R_B_from_F_des, dtype=float)
    contact_B = transform_points(T_B_from_P, contact_P)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = contact_B-rotation @ np.asarray(tip_in_F, dtype=float)
    return pose


def main():
    cal = load_calibration()
    T_BP, T_PB = np.array(cal["T_B_from_P_mm"]), np.array(cal["T_P_from_B_mm"])
    sample = cal["sample"]
    contact_P = np.array(sample["piano_contact_mm"])
    contact_B = transform_points(T_BP, contact_P)
    flange_B = flange_pose_from_contact(T_BP, contact_P, sample["R_B_from_F"], cal["probe_tip_in_F_mm"])
    np.testing.assert_allclose(contact_B, sample["expected_contact_B_mm"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(flange_B[:3, 3], sample["expected_flange_B_mm"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(transform_points(T_PB, contact_B), contact_P, atol=1e-9, rtol=0)
    np.testing.assert_allclose(transform_points(cal["T_B_from_P_m"], contact_P/1000), contact_B/1000, atol=1e-12, rtol=0)
    np.set_printoptions(precision=6, suppress=True)
    print("Selected calibration:", cal["id"])
    print("Fit points:", ", ".join(cal["fitted_point_ids"]))
    print("Piano contact (mm):", contact_P)
    print("Base contact (mm):", contact_B)
    print("Base flange at recorded p2 orientation (mm):", flange_B[:3, 3])
    print("Piano +Y 10 mm -> base displacement (mm):", transform_vectors(T_BP, [0., 10., 0.]))
    print("Offline example verified; no controller SDK is loaded.")


if __name__ == "__main__":
    main()
