#!/usr/bin/env python3
"""Draw the exact fingering-reward target points for one frozen score frame."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def add_sphere(scene, position, radius, color):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Render scene has no room for target annotations")
    import mujoco
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array((radius, 0.0, 0.0)), position, np.eye(3).ravel(), color,
    )
    scene.ngeom += 1


def add_line(scene, start, end, color, width=3.0):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Render scene has no room for target annotations")
    import mujoco
    line = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        line, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
        np.eye(3).ravel(), color,
    )
    mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
    scene.ngeom += 1


def draw_keyboard_map(path, model, data, key_ids, default_targets, finger_targets, assigned,
                      *, heading="Fingering reward target on every piano key", display_keys=None):
    """Draw every target directly on its key in an exact top-plane projection."""
    from PIL import Image, ImageDraw, ImageFont
    from tsensei_v3_2_pianist.preview_fingering import FINGER_COLORS, finger_label, note_name

    width, height = 5200, 1500
    image = Image.new("RGB", (width, height), (18, 23, 32))
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 31)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    draw.text((70, 35), heading, font=title, fill="white")
    draw.text((70, 105), "orange dot = default target | colored ring = assigned finger's actual target",
              font=font, fill=(245, 205, 135))
    draw.text((70, 150), "Each key is labeled NOTE / key index (A0 = key 0, C8 = key 87). Top = -world X; right = +world Y.",
              font=font, fill=(220, 225, 235))

    margin_x, top, bottom = 90, 255, 1410
    display_keys = list(range(88)) if display_keys is None else list(display_keys)
    if not display_keys or min(display_keys) < 0 or max(display_keys) >= 88:
        raise ValueError("display_keys must be a nonempty subset of [0, 87]")
    corners_world = []
    for geom_id in key_ids:
        sx, sy = model.geom_size[geom_id, :2]
        local = np.array(((-sx, -sy, 0), (-sx, sy, 0), (sx, sy, 0), (sx, -sy, 0)))
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        corners_world.append(local @ rotation.T + data.geom_xpos[geom_id])
    all_points = np.concatenate([corners_world[key] for key in display_keys])
    world_y_min, world_y_max = all_points[:, 1].min(), all_points[:, 1].max()
    world_x_min, world_x_max = all_points[:, 0].min(), all_points[:, 0].max()
    pad_x, pad_y = 0.012, 0.012
    world_x_min -= pad_x
    world_x_max += pad_x
    world_y_min -= pad_y
    world_y_max += pad_y

    def pixel(world):
        px = margin_x + (world[1] - world_y_min) / (world_y_max - world_y_min) * (width - 2 * margin_x)
        py = top + (world[0] - world_x_min) / (world_x_max - world_x_min) * (bottom - top)
        return float(px), float(py)

    kinds = ["black" in (model.geom(g).name or "") for g in key_ids]
    # White keys first, then black keys, matching a physical keyboard viewed from above.
    for black_pass in (False, True):
        for key in display_keys:
            geom_id, corners, black = key_ids[key], corners_world[key], kinds[key]
            if black != black_pass:
                continue
            polygon = [pixel(point) for point in corners]
            fill = (30, 32, 38) if black else (241, 241, 236)
            outline = (210, 210, 215) if black else (40, 43, 50)
            draw.polygon(polygon, fill=fill, outline=outline, width=3)

    active_by_key = {int(key): int(finger) for finger, key in np.argwhere(assigned)}
    for key in display_keys:
        geom_id, black = key_ids[key], kinds[key]
        center = pixel(data.geom_xpos[geom_id])
        target = pixel(default_targets[key])
        label_color = (245, 245, 245) if black else (25, 28, 35)
        note = note_name(key)
        text = f"{note}\n{key}"
        box = draw.multiline_textbbox((0, 0), text, font=small, spacing=2, align="center")
        label_y = center[1] - 26 if black else bottom - 150
        draw.multiline_text((center[0] - (box[2] - box[0]) / 2, label_y), text,
                            font=small, fill=label_color, spacing=2, align="center")
        radius = 13
        draw.ellipse((target[0] - radius, target[1] - radius,
                      target[0] + radius, target[1] + radius),
                     fill=(255, 145, 20), outline=(110, 55, 0), width=3)
        if key in active_by_key:
            finger = active_by_key[key]
            target = pixel(finger_targets[finger, key])
            color = tuple((255 * FINGER_COLORS[finger % 5, :3]).astype(int))
            ring = 24
            draw.ellipse((target[0] - ring, target[1] - ring,
                          target[0] + ring, target[1] + ring), outline=color, width=9)
            draw.text((target[0] + 28, target[1] - 20), finger_label(finger),
                      font=font, fill=color, stroke_width=3, stroke_fill=(18, 23, 32))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    outputs = ROOT / "training_mjlab/outputs"
    parser.add_argument("--scene", type=Path,
                        default=outputs / "scene_twinkle_noarms_20hz200hz_20260923")
    parser.add_argument("--task", type=Path,
                        default=outputs / "task_twinkle_noarms_20hz200hz_20260923")
    parser.add_argument("--rollout", type=Path,
                        default=ROOT / "preview/outputs/replays/twinkle_noarms_best17000_20260923/rollout.npz")
    parser.add_argument("--frame", type=int, default=120, help="Zero-based score/rollout frame")
    parser.add_argument("--view", choices=("front", "performer", "overview"), default="front")
    parser.add_argument("--keyboard-map-output", type=Path,
                        help="Optional 5200x1500 plan view with a label and target on every key")
    parser.add_argument("--thumb-score-map-output", type=Path,
                        help="Optional plan view highlighting every key assigned to a thumb in the score")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from tsensei_v3_2_pianist.preview_backend import configure_backend
    configure_backend(headless=True)
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from tsensei_v3_2_pianist.preview_camera import configure_camera
    from tsensei_v3_2_pianist.preview_fingering import (
        FINGER_COLORS, FINGER_NAMES, finger_label, note_name,
    )
    from tsensei_v3_2_pianist.fingering_targets import target_position, options_from_manifest, THUMB_GLOBAL_INDICES

    manifest = json.loads((args.task / "task.json").read_text())
    score = np.load(args.task / "score.npz")
    rollout = np.load(args.rollout)
    if not 0 <= args.frame < len(score["fingers"]):
        parser.error(f"--frame must be in [0, {len(score['fingers']) - 1}]")
    if len(rollout["qpos"]) != len(score["fingers"]):
        parser.error("Rollout and frozen score lengths differ")

    model = mujoco.MjModel.from_xml_path(str(args.scene / "scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = rollout["qpos"][args.frame]
    mujoco.mj_forward(model, data)
    key_ids = np.array([model.geom(name).id for name in manifest["key_geoms"]])
    tip_ids = np.array([model.site(name).id for name in manifest["fingertip_sites"]])

    # This is deliberately identical to PianoTaskEnv.snapshot(): offsets are in
    # world X/Z, not the key's rotated local frame.
    targets = data.geom_xpos[key_ids].copy()
    targets[:, 0] += 0.35 * model.geom_size[key_ids, 0]
    targets[:, 2] += 0.5 * model.geom_size[key_ids, 2]
    finger_targets = np.repeat(targets[None], 10, axis=0)
    if "fingering_target" in manifest:
        task_profile = manifest.get('task_profile', 'tsensei_v1')
        thumb_target = options_from_manifest(manifest['fingering_target'], task_profile)
        for finger in range(10):
            for key, geom_id in enumerate(key_ids):
                finger_targets[finger, key] = target_position(
                    data.geom_xpos[geom_id], model.geom_size[geom_id], model.geom(geom_id).name,
                    thumb=finger in THUMB_GLOBAL_INDICES,
                    task_profile=task_profile, thumb_target=thumb_target,
                )
    assigned = score["fingers"][args.frame] > 0

    width, height = 1600, 900
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    camera = mujoco.MjvCamera()
    camera_settings = {
        "front": dict(distance=1.05, elevation=-34),
        "performer": dict(distance=1.05, elevation=-60),
        "overview": dict(distance=1.25, elevation=-72),
    }[args.view]
    configure_camera(camera, model, view=args.view, framing="full", **camera_settings)
    rows = []
    with mujoco.Renderer(model, width=width, height=height, max_geom=4000) as renderer:
        renderer.update_scene(data, camera=camera)
        scene = renderer.scene
        all_color = np.array((1.0, 0.55, 0.05, 0.75), dtype=np.float32)
        for target in targets:
            add_sphere(scene, target, 0.0022, all_color)
        for finger, key in np.argwhere(assigned):
            color = FINGER_COLORS[finger % 5]
            target = finger_targets[finger, key]
            add_line(scene, data.site_xpos[tip_ids[finger]], target, color)
            add_sphere(scene, target, 0.007, color)
            distance = float(np.linalg.norm(data.site_xpos[tip_ids[finger]] - target))
            rows.append({
                "finger": finger_label(finger),
                "finger_name": FINGER_NAMES[finger % 5],
                "key": int(key),
                "note": note_name(key),
                "target_world_xyz_m": target.tolist(),
                "fingertip_world_xyz_m": data.site_xpos[tip_ids[finger]].tolist(),
                "distance_m": distance,
            })
        image = Image.fromarray(renderer.render())

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    draw.rectangle((0, 0, width, 118), fill=(18, 23, 32))
    draw.text((24, 14), f"Exact fingering-reward targets | score frame {args.frame}",
              font=title, fill="white")
    draw.text((24, 58), "small orange: all 88 key target points | large color: assigned target | line: fingertip distance",
              font=font, fill=(235, 235, 235))
    formula_text = (
        "thumb X = +X key end - 2.0 cm (white) / 1.5 cm (black); other fingers retain default X"
        if "fingering_target" in manifest else
        "target = key geom position + [0.35 x half-length, 0, 0.5 x half-thickness] in world axes"
    )
    draw.text((24, 88), formula_text, font=font, fill=(255, 195, 105))
    y = height - 42 * (len(rows) + 1) - 12
    draw.rectangle((0, y - 10, width, height), fill=(18, 23, 32))
    for row in rows:
        finger = int(row["finger"][1]) - 1
        text = (f"{row['finger']} {row['finger_name']} -> {row['note']} (key {row['key']}), "
                f"distance {100 * row['distance_m']:.1f} cm, target xyz {np.round(row['target_world_xyz_m'], 3)} m")
        draw.text((24, y), text, font=font,
                  fill=tuple((255 * FINGER_COLORS[finger, :3]).astype(int)))
        y += 42

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    if args.keyboard_map_output:
        draw_keyboard_map(args.keyboard_map_output, model, data, key_ids, targets, finger_targets, assigned)
    if args.thumb_score_map_output:
        thumb_assignments = np.zeros((10, 88), dtype=bool)
        thumb_assignments[list(THUMB_GLOBAL_INDICES)] = score["fingers"][:, list(THUMB_GLOBAL_INDICES)].any(axis=0)
        draw_keyboard_map(
            args.thumb_score_map_output, model, data, key_ids, targets, finger_targets,
            thumb_assignments,
            heading="Twinkle Twinkle: every thumb fingering-reward target",
            display_keys=range(
                max(0, int(np.flatnonzero(thumb_assignments.any(axis=0)).min()) - 3),
                min(88, int(np.flatnonzero(thumb_assignments.any(axis=0)).max()) + 4),
            ),
        )
    report = {
        "scene": str(args.scene.resolve()), "task": str(args.task.resolve()),
        "rollout": str(args.rollout.resolve()), "score_frame": args.frame, "view": args.view,
        "fingering_target": manifest.get("fingering_target", "legacy default only"),
        "finger_tolerance_m": manifest["finger_tolerance"], "assignments": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())
    if args.keyboard_map_output:
        print(args.keyboard_map_output.resolve())
    if args.thumb_score_map_output:
        print(args.thumb_score_map_output.resolve())
    for row in rows:
        print(f"{row['finger']} -> {row['note']}: target={row['target_world_xyz_m']}, "
              f"distance={row['distance_m']:.6f} m")


if __name__ == "__main__":
    main()
