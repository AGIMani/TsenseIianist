"""2026-10-25: virtual forearm joint names are now palm_base_*; physical axes unchanged.

Provisional pose; axis conventions inferred from the supplied DXSS reference."""
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
PALM_BASE_DOFS = {
    "palm_base_tx": ("slide", (0, -1, 0), (-0.6, 0.6), 300, 30, 50),
    "palm_base_ty": ("slide", (0, 0, 1), (-0.05, 0.15), 300, 30, 50),
    "palm_base_tz": ("slide", (-1, 0, 0), (-0.025, 0), 1000, 50, 50),
    "palm_base_pitch": ("hinge", (1, 0, 0), (-0.5, 0.5), 50, 3, 5),
    "palm_base_yaw": ("hinge", (0, -1, 0), (-math.pi/6, math.pi/6), 300, 10, 5),
}
DEFAULT_PALM_BASE_DOFS = tuple(PALM_BASE_DOFS)
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
    "palm_base_tx": "Palm_Base_Tx",
    "palm_base_ty": "Palm_Base_Ty",
    "palm_base_tz": "Palm_Base_Tz",
    "palm_base_pitch": "Palm_Base_Pitch",
    "palm_base_yaw": "Palm_Base_Yaw",
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
    "palm_base_tx": 0., "palm_base_ty": 0., "palm_base_tz": 0.,
    "palm_base_pitch": 0., "palm_base_yaw": 0.,
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
           'palm_base_tx': 0.0,
           'palm_base_ty': 0.04,
           'palm_base_tz': 0.0,
           'palm_base_pitch': 0.0,
           'palm_base_yaw': 0.5235987755982988},
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
          'palm_base_tx': 0.0,
          'palm_base_ty': 0.04,
          'palm_base_tz': 0.0,
          'palm_base_pitch': 0.0,
          'palm_base_yaw': 0.5235987755982988}}


def _apply_export_pose_overrides():
    """World XYZ inputs become internal slides; do not change the travel frame.

    Mount XY remains the original installation reference; changing it as well
    would shift the reachable workspace. World Z sets the mount, with Tz=0.
    """
    global RIGHT_POSITION, LEFT_POSITION
    raw = os.environ.get('TSENSEI_V32_EXPORT_POSE')
    if not raw:
        return
    from tsensei_v3_2_pianist.initial_world_pose import validate, slides_for_world
    values = validate(json.loads(raw))
    RIGHT_POSITION = (*RIGHT_POSITION[:2], values['world_z'])
    LEFT_POSITION = (*LEFT_POSITION[:2], values['world_z'])
    for side in ('right', 'left'):
        mount = RIGHT_POSITION if side == 'right' else LEFT_POSITION
        INITIAL_ANGLES_BY_SIDE[side].update(slides_for_world(values, side, mount))
        INITIAL_ANGLES_BY_SIDE[side]['palm_base_yaw'] = values['palm_base_yaw']


_apply_export_pose_overrides()
