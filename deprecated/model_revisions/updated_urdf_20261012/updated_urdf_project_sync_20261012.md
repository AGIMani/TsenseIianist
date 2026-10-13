# Updated URDF project synchronization (2026-10-12)

Source: `/home/hj/hand_middle_dip_standard_description/urdf/hand_middle_dip_standard.urdf`.
Output: `assets/urdf/TsenseI_left_updated_project.urdf`.

The output uses the project's 20-DOF left-hand joint/link contract. The 53-joint source is reduced by removing each MCP/PIP/DIP slider and rod and by folding the corresponding output carrier chain into the project links:

| Project joint | Source motion | Source mount/origin |
|---|---|---|
| `left_hand_{index,mid,ring,pinky}_MCP_joint` | `{finger}_mcp_flexion` | `{finger}_mount_fixed` |
| `left_hand_{index,mid,ring,pinky}_swing_joint` | `{finger}_mcp_swing` | same |
| `left_hand_{index,mid,ring,pinky}_PIP_joint` | `{finger}_pip_flexion` | same |
| `left_hand_{index,mid,ring,pinky}_DIP_joint` | `{finger}_dip_flexion` | same |
| `left_hand_thumb_rota_block_joint` | `thumb_rot1` | `thumb_mount_fixed` |
| `left_hand_thumb_rotaback_joint2` | `thumb_rot2` | same |
| `left_hand_thumb_MCP_joint` | `thumb_f2_flexion` | same |
| `left_hand_thumb_PIP_joint` | `thumb_f1_flexion` | same |

`hand_base` is renamed to the project's `base_link`; all root-level joints now attach to `base_link`, and mesh paths are relative to the project asset directory. Inertials are copied from the active project A3.2 model because the uploaded source URDF is kinematic-only and contains no `<inertial>` elements.

The uploaded file does not contain an explicit homogeneous transform from its `hand_base` frame to the project's canonical base frame. Therefore the conversion establishes the project frame/name contract while preserving the uploaded joint origins and axes. A measured CAD transform is still required before treating absolute placement as hardware-calibrated.

## 2026-10-12 follow-up

The converted model now applies the measured convention requested for this revision: source `+X` (middle-finger direction) maps to project `+Z`, source `+Y` remains the index-to-pinky span, and source `+Z` maps to project `-X`, so project `+X` points toward the palm. The origin is placed at `[0.0237, 0.0065, 0.0441380069844]` in the project's four-hole mounting-point convention.

`left_hand_thumb_rotaback_joint2` is now `[0, 1.74]` rad. Its axis sign is reversed after the frame transform so the positive interval preserves the previous physical rotation direction.

The rota-block link retains the uploaded motor/housing visual set (`NR_G0040.stl`, `NR_G0038_housing.stl`, `NR_G0038_output.stl`, `NR_G0037.stl`, `NR_G0003.stl`, `NR_G0007.stl`, `NR_G0011.stl`) under `assets/meshes_updated/visual/`.

## 2026-10-12 repair

The previous global frame rotation caused the links to separate in the viewer. It was reverted. The active output now uses the uploaded model's original internal link/joint origins and orientations (the stable assembled pose), while keeping the project link names, the positive `left_hand_thumb_rotaback_joint2` interval `[0, 1.74]` with axis `0 0 -1`, and the complete rota-block motor visuals. The experimental global-frame script was moved under `deprecated/tools/` and is not part of the active workflow.

## 2026-10-12 motor-visual repair

The motor was in the source `thumb_mount` link, which had been omitted when the auxiliary chain was folded. Its fixed visual geometry is now merged into `left_hand_thumb_rota_block_link`; `thumb_rot1` has zero origin, so the copied visual origin is unchanged in the rota-block frame. The missing `NR_G0036.stl` motor body is now explicitly referenced. Existing shared housing/gear visuals are deduplicated by filename.
