"""Build MJCF directly from bundled URDF, without ROS or external model paths."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from dm_control import mjcf
from tsensei_v3_2_pianist.models.hands.tsensei_hand_constants import (
    PROVISIONAL_FINGER_EFFORT, PROVISIONAL_FINGER_ARMATURE,
    PROVISIONAL_FINGER_KP, control_label)


def _origin(element):
    origin = element.find("origin")
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    rotation = Rotation.from_euler("xyz", rpy)
    return xyz, rotation


def build(path, name):
    source = ET.parse(path).getroot()
    model = mjcf.RootElement(model=name)
    model.compiler.angle = "radian"
    from tsensei_v3_2_pianist.physics_settings import load
    settings = load()["hand"]
    from tsensei_v3_2_pianist.models.hands import collision_proxies
    proxies = collision_proxies.load()
    side = name.strip("_")
    for attr in ("friction", "solref", "solimp", "priority"):
        setattr(model.default.geom, attr, settings[attr])
    # TODO(V32-04): Calibrate servo kp and damping; URDF contains no servo gains.
    model.default.joint.damping = .05
    links = {link.get("name"): link for link in source.findall("link")}
    children = {}
    child_names = set()
    for joint in source.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
        child_names.add(joint.find("child").get("link"))
    roots = set(links) - child_names
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root, got {roots}")

    def add_link(link_name, parent, joint=None):
        link = links[link_name]
        kwargs = {"name": link_name}
        if joint is not None:
            xyz, rotation = _origin(joint)
            kwargs.update(pos=xyz, quat=np.roll(rotation.as_quat(), 1))
        body = parent.add("body", **kwargs)
        inertial = link.find("inertial")
        if inertial is not None:
            xyz, rotation = _origin(inertial)
            i = inertial.find("inertia").attrib
            tensor = np.array([[float(i['ixx']),float(i['ixy']),float(i['ixz'])],
                               [float(i['ixy']),float(i['iyy']),float(i['iyz'])],
                               [float(i['ixz']),float(i['iyz']),float(i['izz'])]])
            rot = rotation.as_matrix()
            tensor = rot @ tensor @ rot.T
            body.add("inertial", pos=xyz, mass=float(inertial.find("mass").get("value")),
                     fullinertia=tensor[(0,1,2,0,0,1),(0,1,2,1,2,2)])
        if joint is not None and joint.get("type") != "fixed":
            if joint.get("type") != "revolute" or joint.find("mimic") is not None:
                raise ValueError(f"Unsupported URDF joint: {joint.attrib}")
            limits = joint.find("limit")
            bounds = (float(limits.get("lower")), float(limits.get("upper")))
            element = body.add("joint", name=joint.get("name"), type="hinge",
                               axis=np.fromstring(joint.find("axis").get("xyz"), sep=" "),
                               range=bounds, limited=True,
                               armature=PROVISIONAL_FINGER_ARMATURE)
            effort = float(limits.get("effort"))
            if effort <= 0:
                # TODO(V32-04): Zero is an uncalibrated CAD-export placeholder.
                effort = PROVISIONAL_FINGER_EFFORT
            model.actuator.add("position", name=control_label(name, joint.get("name")), joint=element,
                               kp=PROVISIONAL_FINGER_KP, ctrlrange=bounds, ctrllimited=True,
                               forcerange=(-effort, effort), forcelimited=True)
            # TODO(V32-05): URDF velocity is recorded in the source; no speed limiter yet.
        for kind in ("visual", "collision"):
            for idx, geom in enumerate(link.findall(kind)):
                if kind == "collision" and proxies is not None:
                    attributes = collision_proxies.for_link(proxies, side, link_name)
                    if attributes is not None:
                        # A compound proxy supplies the complete link collider.
                        # Remaining URDF collision entries serve direct URDF consumers.
                        if idx != 0:
                            continue
                        components = attributes if isinstance(attributes, list) else [attributes]
                        for part_index, part in enumerate(components):
                            part = dict(part)
                            if part["type"] == "mesh" and "_mesh_file" in part:
                                mesh_file = part.pop("_mesh_file")
                                suffix = f"_{part_index}" if len(components) > 1 else ""
                                mesh_name = f"collision_proxy_{side}_{link_name}{suffix}"
                                asset = model.find("mesh", mesh_name)
                                if asset is None:
                                    asset = model.asset.add("mesh", name=mesh_name, file=mesh_file)
                                xyz, rotation = _origin(geom)
                                body.add("geom", name=f"{link_name}_{kind}_{part_index}",
                                         type="mesh", mesh=asset, pos=xyz,
                                         quat=np.roll(rotation.as_quat(), 1),
                                         contype=1, conaffinity=1, group=3,
                                         rgba=(.5, .5, .5, .3))
                            else:
                                body.add("geom", name=f"{link_name}_{kind}_{part_index}",
                                         contype=1, conaffinity=1, group=3,
                                         rgba=(.5, .5, .5, .3), **part)
                        continue
                mesh = geom.find("geometry/mesh")
                if mesh is None:
                    raise ValueError(f"Expected mesh geometry: {link_name}/{kind}")
                file = (path.parent / mesh.get("filename")).resolve()
                if file.with_suffix(".obj").is_file():
                    file = file.with_suffix(".obj")
                if not file.is_relative_to(path.parent.parent.resolve()):
                    raise ValueError(f"Asset escapes the standalone package: {file}")
                mesh_name = file.stem
                asset = model.find("mesh", mesh_name)
                if asset is None:
                    asset = model.asset.add("mesh", name=mesh_name, file=str(file),
                                            scale=np.fromstring(mesh.get("scale", "1 1 1"), sep=" "))
                xyz, rotation = _origin(geom)
                visual = kind == "visual"
                # TODO(V32-06): MuJoCo uses convex mesh collisions; validate/decompose CAD.
                body.add("geom", name=f"{link_name}_{kind}_{idx}", type="mesh", mesh=asset,
                         pos=xyz, quat=np.roll(rotation.as_quat(), 1),
                         contype=0 if visual else 1, conaffinity=0 if visual else 1,
                         group=1 if visual else 3, rgba=(.75,.78,.85,1) if visual else (.5,.5,.5,.3))
        for child in children.get(link_name, []):
            add_link(child.find("child").get("link"), body, child)
        return body

    root = add_link(roots.pop(), model.worldbody)
    return model, root
