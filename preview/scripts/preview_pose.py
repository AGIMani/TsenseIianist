#!/usr/bin/env python3
# 2026-10-25: External initial position offsets use world-frame metres; palm-base yaw remains a joint angle in radians.
"""Preview V3.2: supplied left hand and a provisional mirrored right hand."""
import argparse
import json
import math
import os
import time
import sys
from pathlib import Path


# Resolve this standalone project even when launched from another directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "preview" / "outputs" / "pose.png"
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    startup_begin = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-hz', type=float, default=20., help='Control frequency (default: 20 Hz, current training scene)')
    parser.add_argument('--physics-hz', type=float, default=200., help='Physics frequency (default: 200 Hz, current training scene)')
    parser.add_argument('--hand-collision-detection-only', action=argparse.BooleanOptionalAction, default=False,
                        help='Optional legacy mode: detect same/cross-hand contacts without forces (default: physical)')
    parser.add_argument('--self-collision-detection-only', action=argparse.BooleanOptionalAction, default=False,
                        help='Optional legacy mode: detect same-hand contacts without forces (default: physical)')
    parser.add_argument('--arms', action=argparse.BooleanOptionalAction, default=True,
                        help='Show white M6 arms following the hand wrists by differential IK (preview only)')
    parser.add_argument('--show-forearm-collisions', action='store_true', help='Color exact hand and M6 reward collision hulls')
    parser.add_argument('--collision-profile', choices=('shadow_primitive_v1', 'convex_hull_v1', 'legacy_mesh_v1'),
                        help='Select hand collision geometry for this preview; convex_hull_v1 is an opt-in full-link hull audit')
    parser.add_argument('--real-collision-mesh', action='store_true',
                        help='Use the original URDF collision meshes for this preview (overrides the primitive profile)')
    parser.add_argument('--show-hand-collisions', action='store_true',
                        help='Show hand collision geoms as translucent orange overlays; real-collision-mesh enables this automatically')
    parser.add_argument('--arm-config', type=Path, help='Shoulder mounting and wrist adapter JSON')
    parser.add_argument('--hand-connector', action=argparse.BooleanOptionalAction, default=True,
                        help='Show the supplied STEP connector on both hands (preview only)')
    parser.add_argument('--connector-config', type=Path, help='Connector alignment and saved per-hand offsets JSON')
    parser.add_argument('--connector-controls', action=argparse.BooleanOptionalAction, default=True,
                        help='Interactive XYZ (mm), Roll/Pitch/Yaw (deg), reset and save controls')
    parser.add_argument("--contact-controls", action=argparse.BooleanOptionalAction, default=True, help="Show live solimp sliders in a companion window")
    parser.add_argument("--gpu", choices=("auto", "nvidia", "system"), default="auto", help="Select NVIDIA PRIME automatically on Linux, force NVIDIA, or keep system graphics")
    parser.add_argument("--full-sync", action="store_true", help="Use legacy full-model viewer sync for diagnostics")
    parser.add_argument("--duration", type=float, help="Close the interactive viewer after this many seconds")
    parser.add_argument("--profile-output", type=Path, help="Write graphics backend and measured timings as JSON")
    parser.add_argument("--paused", action="store_true", help="Start the interactive simulation paused; Space resumes")
    parser.add_argument("--headless", action="store_true", help="Render a PNG instead of opening a window")
    parser.add_argument("--show-pad-collisions", action="store_true", help="Overlay the distal collision convex hulls as green wireframes")
    parser.add_argument("--show-fingertips", action="store_true", help="Show black spheres centered at the provisional distal-face landmarks")
    parser.add_argument("--show-coordinate-frames", action="store_true", help="Show world and both palm_base XYZ axes (hidden by default)")
    parser.add_argument("--axis-length", type=float, default=.09, help="World and palm_base axis length in metres (default: 0.09)")
    parser.add_argument("--transparent-model", action="store_true", help="Make moving bodies translucent to reveal occluded coordinate axes")
    parser.add_argument("--include-assembly-collisions", action="store_true", help="Disable the two provisional palm/thumb collision exclusions (V32-06)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--world-z-offset", type=float, default=0, help="Adjust both mounting positions along world Z, metres")
    parser.add_argument("--world-x-offset", type=float, default=0, help="Adjust both mounting positions along world X, metres")
    parser.add_argument("--settle-steps", type=int, default=0, help="Optional physics steps under neutral control")
    parser.add_argument("--save-initial-pose", action="store_true", help="Write current pose to hand constants after the window closes, or after headless rendering; P saves during interaction")
    parser.add_argument("--save-pose-source", choices=("ctrl", "qpos"), default=None, help="Save actual positions (default with arms) or Control targets (default without arms)")
    parser.add_argument("--export-xml", type=Path, help="Export scene XML and all assets to this directory")
    if any(value.split('=', 1)[0] in ('--height-offset', '--depth-offset') for value in sys.argv[1:]):
        parser.error('Removed ambiguous offsets; use --world-z-offset and --world-x-offset in metres')
    args = parser.parse_args()
    for hz in (args.control_hz, args.physics_hz):
        if not math.isfinite(hz) or hz <= 0:
            parser.error('--control-hz and --physics-hz must be finite and positive')
    if not math.isclose(args.physics_hz / args.control_hz, round(args.physics_hz / args.control_hz), rel_tol=0, abs_tol=1e-9):
        parser.error('physics frequency must be an integer multiple of control frequency')
    args.save_pose_source = args.save_pose_source or ("qpos" if args.arms else "ctrl")
    if not math.isfinite(args.axis_length) or args.axis_length <= 0:
        parser.error("--axis-length must be a finite positive number")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be a finite positive number")
    if args.headless and (args.duration is not None or args.full_sync):
        parser.error("--duration and --full-sync require the interactive window")
    if args.show_forearm_collisions and not args.arms:
        parser.error('--show-forearm-collisions requires arms')
    if args.real_collision_mesh and args.collision_profile is not None:
        parser.error('--real-collision-mesh cannot be combined with --collision-profile')
    if args.real_collision_mesh:
        os.environ['TSENSEI_V32_COLLISION_PROFILE'] = 'legacy_mesh_v1'
    elif args.collision_profile is not None:
        os.environ['TSENSEI_V32_COLLISION_PROFILE'] = args.collision_profile
    collision_profile = os.environ.get('TSENSEI_V32_COLLISION_PROFILE', 'shadow_primitive_v1')
    show_hand_collisions = args.show_hand_collisions or args.real_collision_mesh or collision_profile == 'convex_hull_v1'
    if args.headless and args.paused:
        parser.error("--paused is only available for the interactive window")
    if args.settle_steps < 0:
        parser.error("--settle-steps must be nonnegative")
    from tsensei_v3_2_pianist.preview_backend import configure_backend, probe_desktop_renderer, current_gl_info
    configure_backend(headless=args.headless, gpu=args.gpu)
    import numpy as np
    import mujoco
    from dm_control import mjcf
    from tsensei_v3_2_pianist import suite
    from tsensei_v3_2_pianist.models.hands import tsensei_hand_constants as consts
    from tsensei_v3_2_pianist.preview_geometry import PadCollisionWireframe, color_hand_collision_geoms, configure_options
    from tsensei_v3_2_pianist.preview_frames import CoordinateFrames
    from tsensei_v3_2_pianist.preview_pose_config import InitialPoseSaver

    graphics = None if args.headless else probe_desktop_renderer()
    if graphics is not None:
        print("OpenGL renderer: " + graphics['renderer'], flush=True)
        if graphics['software_renderer']:
            print("Software rendering detected. Try --gpu nvidia on an NVIDIA desktop.", flush=True)
    model_begin = time.perf_counter()
    env = suite.load(task_kwargs={"change_color_on_activation": True,
                                 "gravity_compensation": True,
                                 "control_timestep": 1 / args.control_hz,
                                 "physics_timestep": 1 / args.physics_hz,
                                 "exclude_assembly_collisions": not args.include_assembly_collisions})
    print(f'Hand collision profile: {collision_profile}', flush=True)
    print('Hand gravity compensation: on', flush=True)
    connector_preview = None
    if args.hand_connector:
        from tsensei_v3_2_pianist.preview_connector import ConnectorPreview, DEFAULT_CONFIG as CONNECTOR_CONFIG, tool_transforms
        connector_preview = ConnectorPreview(env.task.root_entity.mjcf_model, args.connector_config or CONNECTOR_CONFIG)
    arm_preview = None
    if args.arms:
        from tsensei_v3_2_pianist.preview_arm_ik import ArmPreview, DEFAULT_CONFIG
        arm_preview = ArmPreview(env.task.root_entity.mjcf_model, args.arm_config or DEFAULT_CONFIG,
                                 tool_overrides=tool_transforms(connector_preview.config) if connector_preview else None)
        for hand in (env.task.right_hand, env.task.left_hand):
            for geom in hand.mjcf_model.find_all('geom'):
                if geom.group == 1:
                    geom.rgba = (.94, .95, .97, 1)
    for hand in (env.task.right_hand, env.task.left_hand):
        pos = np.array(hand.root_body.pos)
        pos += (args.world_x_offset, 0, args.world_z_offset)
        hand.root_body.pos = pos
    if args.self_collision_detection_only or args.hand_collision_detection_only:
        from tsensei_v3_2_pianist.detection_only_collision import apply_to_mjcf
        count = apply_to_mjcf(env.task.root_entity.mjcf_model, include_cross_hand=args.hand_collision_detection_only)
        print(f'Detection-only hand contact pairs: {count}; cross-hand={args.hand_collision_detection_only}')
    else:
        print('Physical contacts enabled: same-hand, cross-hand, hand-piano', flush=True)
    env.reset()
    physics = env.physics
    if show_hand_collisions:
        color_hand_collision_geoms(physics.model.ptr)
    for _ in range(args.settle_steps):
        physics.step()
    model_ready = time.perf_counter()
    hands = (env.task.right_hand, env.task.left_hand)
    if arm_preview is not None:
        arm_preview.bind(physics, hands)
        arm_preview.show_collision_regions = args.show_forearm_collisions
        if args.show_forearm_collisions:
            from tsensei_v3_2_pianist.preview_arm_collision import color_regions
            color_regions(physics.model.ptr, arm_preview.collision)
        physics.model.ptr.vis.headlight.ambient[:] = (.45, .45, .45)
    if connector_preview is not None:
        connector_preview.bind(physics, hands, arm_preview)
    pose_saver = InitialPoseSaver(
        physics, hands, consts.__file__, PROJECT_ROOT / 'preview/outputs/initial_pose_backups',
        source=args.save_pose_source)
    print(f'Initial pose config: {pose_saver.path}; save source: {args.save_pose_source}', flush=True)
    wireframe = PadCollisionWireframe(physics, hands) if args.show_pad_collisions else None
    coordinate_frames = (CoordinateFrames(physics, hands, args.axis_length, arm_preview=arm_preview)
                         if args.show_coordinate_frames else None)
    performance = {"gpu_selection": args.gpu, "graphics": graphics,
                   "model_load_and_reset_s": model_ready - model_begin,
                   "startup_s": time.perf_counter() - startup_begin,
                   "model_bytes": mujoco.mj_sizeModel(physics.model.ptr),
                   "mesh_triangles": int(physics.model.ptr.nmeshface)}
    from tsensei_v3_2_pianist.physics_settings import from_model, shadow_comparison, SolimpController
    compiled_contacts = from_model(physics.model.ptr, env.control_timestep())
    shadow_contact = shadow_comparison(compiled_contacts)
    contacts = SolimpController(physics.model.ptr, compiled_contacts)
    state = 'MATCH' if shadow_contact['matches_shadow'] else 'DIFF'
    effective = shadow_contact['hand_key_contact']
    print(f'Shadow contact parameters: {state}; hand-key solref={effective["mixed_solref"]}, '
          f'solimp={effective["mixed_solimp"]}, refsafe time constant='
          f'{effective["refsafe_timeconst_s"]:.6g}s', flush=True)
    report = {
        "gravity_compensation": True,
        "collision_profile": collision_profile,
        "real_collision_mesh": collision_profile == "legacy_mesh_v1",
        "show_hand_collisions": show_hand_collisions,
        "contact_settings": contacts.settings,
        "shadow_contact_comparison": shadow_contact,
        "graphics": graphics,
        "capture": "before_interaction" if not args.headless else "headless_snapshot",
        "interactive_start_paused": args.paused,
        "initial_pose_config": str(pose_saver.path),
        "save_pose_source": args.save_pose_source,
        "model_version": "TsenseI_A3.2",
        "mounting_frame": "normalized V3-style frame, not the original V3.2 CAD base frame",
        "canonical_from_source_root": json.loads(
            (consts.ASSETS / "model_derivation.json").read_text())["canonical_from_source_root"],
        "right_hand_source": "provisional mirror of the supplied left hand; TODO(V32-10)",
        "finger_effort_nm": consts.PROVISIONAL_FINGER_EFFORT,
        "finger_effort_is_provisional": True,
        "finger_armature_kg_m2": consts.PROVISIONAL_FINGER_ARMATURE,
        "finger_armature_is_provisional": True,
        "finger_kp_nm_per_rad": consts.PROVISIONAL_FINGER_KP,
        "contact_response": {
            "same_hand": "detection_only" if (args.self_collision_detection_only or args.hand_collision_detection_only) else "physical",
            "cross_hand": "detection_only" if args.hand_collision_detection_only else "physical",
            "hand_piano": "physical",
        },
        "assembly_collision_exclusions": not args.include_assembly_collisions,
        "pose": "reset" if not args.settle_steps else "neutral_control_after_physics_steps",
        "settle_steps": args.settle_steps,
        "show_pad_collisions": args.show_pad_collisions,
        "show_fingertips": args.show_fingertips,
        "show_coordinate_frames": args.show_coordinate_frames,
        "transparent_model": args.transparent_model,
        "coordinate_frame_definition": "World origin, normalized wrist base_link frames, and (when arms are enabled) moving M6 J1-J7 child-link frames; RGB = +X,+Y,+Z",
        "coordinate_frames": coordinate_frames.report() if coordinate_frames else [],
        "collision_wireframe_geoms": wireframe.geom_names if wireframe else [],
        "collision_wireframe_edges": wireframe.edge_count if wireframe else 0,
        "nq": physics.model.nq, "nv": physics.model.nv, "nu": physics.model.nu,
        "action_size_including_sustain": env.action_spec().shape[0],
        "contact_count": physics.data.ncon,
        "arms": arm_preview.report() if arm_preview is not None else None,
        "connector": connector_preview.report() if connector_preview is not None else None,
        "penetrations": [{"geom1": physics.model.id2name(int(c.geom1), "geom"),
                          "geom2": physics.model.id2name(int(c.geom2), "geom"),
                          "distance_m": float(c.dist)} for c in physics.data.contact if c.dist < -1e-5],
        "hands": {},
    }
    for hand in (env.task.right_hand, env.task.left_hand):
        report["hands"][hand.name] = {
            "root_position": physics.bind(hand.root_body).xpos.tolist(),
            "root_quaternion_wxyz": physics.bind(hand.root_body).xquat.tolist(),
            "fingertips_world_xyz": physics.bind(hand.fingertip_sites).xpos.tolist(),
            "fingertip_definition": "Provisional midpoint of the bounding box of collision vertices maximizing local +Z; TODO(V32-07): calibrate real contact pad",
            "fingertips_local": [{"finger": site.name.removesuffix("_tip"),
                                  "frame": body.name, "position_m": list(site.pos)}
                                 for body, site in zip(hand.fingertip_bodies, hand.fingertip_sites)],
            "action_names": [a.name for a in hand.actuators],
            "neutral_control": hand.neutral_control.tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if args.export_xml:
        args.export_xml.mkdir(parents=True, exist_ok=True)
        mjcf.export_with_assets(env.task.root_entity.mjcf_model, str(args.export_xml), "scene.xml")
    if args.headless:
        from PIL import Image
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0.03, 0, .06)
        camera.distance = 1.05
        camera.azimuth = 155
        camera.elevation = -50
        if arm_preview is not None:
            arm_preview.camera(camera)
        option = mujoco.MjvOption()
        configure_options(option, args.show_fingertips, args.transparent_model, show_hand_collisions)
        if args.show_forearm_collisions:
            option.geomgroup[4] = 1
            option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
        max_geom = (1000 + (wireframe.edge_count if wireframe else 0)
                    + (coordinate_frames.geom_count if coordinate_frames else 0))
        renderer_begin = time.perf_counter()
        with mujoco.Renderer(physics.model.ptr, height=900, width=1400, max_geom=max_geom) as renderer:
            performance['renderer_init_s'] = time.perf_counter() - renderer_begin
            performance['graphics'] = current_gl_info()
            print("OpenGL renderer: " + performance['graphics']['renderer'], flush=True)
            render_begin = time.perf_counter()
            renderer.update_scene(physics.data.ptr, camera=camera, scene_option=option)
            if wireframe:
                wireframe.append_to_scene(renderer.scene)
            if coordinate_frames:
                coordinate_frames.append_to_scene(renderer.scene)
            from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
            ActiveKeyHighlight(physics.model.ptr).color_render_scene(renderer.scene, physics.data.ptr)
            Image.fromarray(renderer.render()).save(args.output)
            performance["render_and_save_s"] = time.perf_counter() - render_begin
        print(f"Saved {args.output}")
    else:
        from tsensei_v3_2_pianist.preview_runtime import run_interactive
        performance['interactive'] = run_interactive(
            physics, wireframe=wireframe, coordinate_frames=coordinate_frames,
            show_fingertips=args.show_fingertips,
            start_paused=args.paused, duration=args.duration, full_sync=args.full_sync,
            transparent_model=args.transparent_model, show_hand_collisions=show_hand_collisions, on_save=pose_saver.save,
            contact_controller=contacts if args.contact_controls else None,
            arm_preview=arm_preview, connector_preview=connector_preview,
            connector_controls=args.connector_controls)
    if args.save_initial_pose:
        pose_saver.save()
    report['contact_settings'] = contacts.settings
    report['shadow_contact_comparison'] = shadow_comparison(contacts.settings)
    report['graphics'] = performance['graphics']
    if arm_preview is not None:
        report['arms'] = arm_preview.report()
    if connector_preview is not None:
        report['connector'] = connector_preview.report()
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if args.profile_output:
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(json.dumps(performance, indent=2) + "\n")
        print(f"Saved performance report: {args.profile_output}")
    env.close()


if __name__ == "__main__":
    main()
