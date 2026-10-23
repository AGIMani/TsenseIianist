"""Provisional pose; axis conventions inferred from the supplied DXSS reference."""
from pathlib import Path
import json
import math
import os

ASSETS = Path(__file__).resolve().parents[2] / "assets"
# TODO(V32-04): All supplied effort limits are zero. Provisional simulation
# limit follows the upper end of the V2 export (0.6 N m), not measured V3.2 data.
PROVISIONAL_FINGER_EFFORT = .6
# Shadow's reflected motor inertia stabilizes the light CAD finger links.
# TODO(V32-04): simulation adaptation, not measured TsenseI rotor/gear inertia.
# Applies to the 40 finger hinges only, not virtual wrists or piano keys.
PROVISIONAL_FINGER_ARMATURE = .0002  # kg m^2
# Within Shadow's finger gain range (0.5--1.5); fixed-target collision sweeps
# select 1 instead of the original uncalibrated 3. Wrist gains are separate.
PROVISIONAL_FINGER_KP = 1.0  # N m / rad; TODO(V32-04): hardware calibration
# MuJoCo w,x,y,z: local X -> world -Z, Y -> -Y, Z -> -X.
ATTACHMENT_QUATERNION = (0.0, 2**-0.5, 0.0, -(2**-0.5))
# V32-01: simulation mounting is provisionally accepted (user, 2026-09-17).
# Hardware measurement remains a deployment follow-up, not a GPU migration blocker.
RIGHT_POSITION = (0.18, 0.15, 0.10)
LEFT_POSITION = (0.18, -0.15, 0.10)
# type, local axis, range, kp, damping, force limit. Virtual positioning stage.
# TODO(V32-02): Replace virtual wrist with the actual arm/stage and its limits.
# Axes below are for the right hand. The left hand reverses pitch only:
# pitch R:+X / L:-X; yaw R:-Y / L:-Y (equal inputs give same yaw / opposite pitch).
FOREARM_DOFS = {
    "forearm_tx": ("slide", (0, -1, 0), (-0.6, 0.6), 300, 30, 50),
    "forearm_ty": ("slide", (0, 0, 1), (-0.05, 0.15), 300, 30, 50),
    "forearm_tz": ("slide", (-1, 0, 0), (-0.025, 0), 1000, 50, 50),
    "forearm_pitch": ("hinge", (1, 0, 0), (0.0, 0.5), 50, 3, 5),
    "forearm_yaw": ("hinge", (0, -1, 0), (-math.pi/6, math.pi/6), 300, 10, 5),
}
DEFAULT_FOREARM_DOFS = tuple(FOREARM_DOFS)
FINGERTIP_COLORS = ((.8,.2,.8), (.8,.2,.2), (.2,.8,.8), (.2,.2,.8), (.8,.8,.2))
FINGERS = ("thumb", "index", "mid", "ring", "pinky")

# Short actuator names are shown in MuJoCo's narrow Control panel. Joint and
# body names follow the normalized URDF; source-name changes are recorded in
# assets/model_derivation.json (the user-requested pinky naming convention).
CONTROL_LABELS = {
    "thumb_rota_block": "Thumb_Rot",
    "thumb_rotaback_joint2": "Thumb_Rot2",
    "thumb_MCP": "Thumb_MCP",
    "thumb_PIP": "Thumb_PIP",
    "index_MCP": "Index_MCP",
    "index_swing": "Index_SW",
    "index_PIP": "Index_PIP",
    "index_DIP": "Index_DIP",
    "mid_MCP": "Mid_MCP",
    "mid_swing": "Mid_SW",
    "mid_PIP": "Mid_PIP",
    "mid_DIP": "Mid_DIP",
    "ring_MCP": "Ring_MCP",
    "ring_swing": "Ring_SW",
    "ring_PIP": "Ring_PIP",
    "ring_DIP": "Ring_DIP",
    "pinky_MCP": "Pinky_MCP",
    "pinky_swing": "Pinky_SW",
    "pinky_PIP": "Pinky_PIP",
    "pinky_DIP": "Pinky_DIP",
    "forearm_tx": "Palm_Base_Tx",
    "forearm_ty": "Palm_Base_Ty",
    "forearm_tz": "Palm_Base_Tz",
    "forearm_pitch": "Palm_Base_Pitch",
    "forearm_yaw": "Palm_Base_Yaw",
}


def control_label(side, joint_name):
    """Return a compact, unique actuator label such as ``R_Index_SW``."""
    side_name = side.name.lower() if hasattr(side, "name") else str(side).strip("_").lower()
    side_letter = side_name[0].upper()
    source_prefix = side_name + "_hand_"
    short_joint = joint_name.removeprefix(source_prefix).removesuffix("_joint")
    try:
        return f"{side_letter}_{CONTROL_LABELS[short_joint]}"
    except KeyError as error:
        raise ValueError(f"No Control label for joint {joint_name!r}") from error
# TODO(V32-03): Verify these lightly flexed targets, including the redesigned V3.2 thumb base.
INITIAL_ANGLES = {
    # Wrist slide values are metres; wrist/finger hinge values are radians.
    "forearm_tx": 0., "forearm_ty": 0., "forearm_tz": 0.,
    "forearm_pitch": 0., "forearm_yaw": 0.,
    "thumb_rota_block": 1.0, "thumb_rotaback_joint2": .4, "thumb_MCP": .25, "thumb_PIP": .35,
    "index_MCP": .22, "index_PIP": .35, "index_DIP": .18,
    "mid_MCP": .25, "mid_PIP": .38, "mid_DIP": .20,
    "ring_MCP": .28, "ring_PIP": .42, "ring_DIP": .22,
    # New A3.2 order: flexion, lateral MCP, flexion, distal flexion.
    "pinky_MCP": .32, "pinky_swing": 0., "pinky_PIP": .45, "pinky_DIP": .24,
}

# Preview P saves independent left/right initial joint values here. These
# override INITIAL_ANGLES; clear a side's dictionary to use the shared defaults.
# Installation position stays in RIGHT_POSITION / LEFT_POSITION above; wrist
# translations remain joint values, so they must not also be added there.
# 2026-10-20: retain the accepted numeric reference and user-requested old
# Thumb_Rot2 axis/positive angle. New geometry is not asserted coplanar.
# Finger angles only; wrists/mounts unchanged. TODO(V32-03), TODO(V32-07): provisional
# human-like pose and mesh landmarks, not calibrated physical contact pads.
# See docs/initial_five_tip_plane_20261001.md.
INITIAL_ANGLES_BY_SIDE = {'right': {'thumb_rota_block': 1.135929529,
           'thumb_rotaback_joint2': 0.476648962,
           'thumb_MCP': 0.241250147,
           'thumb_PIP': 0.34598514,
           'index_MCP': 0.215778546714,
           'index_swing': 0.0,
           'index_PIP': 0.345984067635,
           'index_DIP': 0.177049848625,
           'mid_MCP': 0.16688924586,
           'mid_swing': 0.0,
           'mid_PIP': 0.33724412436,
           'mid_DIP': 0.179382869067,
           'ring_MCP': 0.1741387815,
           'ring_swing': 0.0,
           'ring_PIP': 0.366567845739,
           'ring_DIP': 0.193132156912,
           'pinky_MCP': 0.193021298123,
           'pinky_swing': 0.0,
           'pinky_PIP': 0.389615614939,
           'pinky_DIP': 0.213344010776,
           'forearm_tx': 0.0,
           'forearm_ty': 0.04,
           'forearm_tz': 0.0,
           'forearm_pitch': 0.0,
           'forearm_yaw': 0.5235987755982988},
 'left': {'thumb_rota_block': 1.135929529,
          'thumb_rotaback_joint2': 0.476648962,
          'thumb_MCP': 0.241250147,
          'thumb_PIP': 0.34598514,
          'index_MCP': 0.215778546714,
          'index_swing': 0.0,
          'index_PIP': 0.345984067635,
          'index_DIP': 0.177049848625,
          'mid_MCP': 0.16688924586,
          'mid_swing': 0.0,
          'mid_PIP': 0.33724412436,
          'mid_DIP': 0.179382869067,
          'ring_MCP': 0.1741387815,
          'ring_swing': 0.0,
          'ring_PIP': 0.366567845739,
          'ring_DIP': 0.193132156912,
          'pinky_MCP': 0.193021298123,
          'pinky_swing': 0.0,
          'pinky_PIP': 0.389615614939,
          'pinky_DIP': 0.213344010776,
          'forearm_tx': 0.0,
          'forearm_ty': 0.04,
          'forearm_tz': 0.0,
          'forearm_pitch': 0.0,
          'forearm_yaw': 0.5235987755982988}}


def _apply_export_pose_overrides():
    """Apply process-local export overrides without editing the source defaults."""
    global RIGHT_POSITION, LEFT_POSITION
    raw = os.environ.get('TSENSEI_V32_EXPORT_POSE')
    if not raw:
        return
    values = json.loads(raw)
    if not isinstance(values, dict) or set(values) - {'height', 'ty', 'yaw'}:
        raise ValueError('Invalid TSENSEI_V32_EXPORT_POSE keys')
    if not values:
        raise ValueError('Empty TSENSEI_V32_EXPORT_POSE override')
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid export pose {name}: {value!r}')
    if 'height' in values:
        if values['height'] <= 0:
            raise ValueError('Export mounting height must be positive')
        RIGHT_POSITION = (*RIGHT_POSITION[:2], float(values['height']))
        LEFT_POSITION = (*LEFT_POSITION[:2], float(values['height']))
    for side in ('right', 'left'):
        if 'ty' in values:
            lo, hi = FOREARM_DOFS['forearm_ty'][2]
            if not lo <= values['ty'] <= hi:
                raise ValueError(f'Export TY must lie in [{lo}, {hi}]')
            INITIAL_ANGLES_BY_SIDE[side]['forearm_ty'] = float(values['ty'])
        if 'yaw' in values:
            lo, hi = FOREARM_DOFS['forearm_yaw'][2]
            if not lo <= values['yaw'] <= hi:
                raise ValueError(f'Export Yaw must lie in [{lo}, {hi}]')
            INITIAL_ANGLES_BY_SIDE[side]['forearm_yaw'] = float(values['yaw'])


_apply_export_pose_overrides()
