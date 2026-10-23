"""Explicit collision profiles for the bundled V3.2 hand meshes.

Visuals, inertia, joints and fingertip landmarks remain in their URDF frames.
The JSON records engineering approximations, not measured hardware geometry.
"""
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_PROFILE = "convex_hull_v1"
# The current profile supports convex components for concave CAD assemblies.
PROFILES = (DEFAULT_PROFILE, "legacy_mesh_v1")
# The previous name remains readable only as a hull-only alias for old saved
# recipes; it no longer contains primitive proxies.
OPTIONAL_PROFILES = ("shadow_primitive_v1",)
_SUPPORTED_PROFILES = PROFILES + OPTIONAL_PROFILES
CONFIG_PATH = Path(__file__).resolve().parents[2] / "assets" / "collision_proxies.json"
HULL_CONFIG_PATH = CONFIG_PATH.with_name("collision_proxies_convex_hull.json")


def load(profile=None):
    profile = profile or os.environ.get("TSENSEI_V32_COLLISION_PROFILE", DEFAULT_PROFILE)
    if profile not in _SUPPORTED_PROFILES:
        raise ValueError(f"Unknown hand collision profile: {profile!r}")
    if profile == "legacy_mesh_v1":
        return None
    config_path = HULL_CONFIG_PATH if profile == "convex_hull_v1" else CONFIG_PATH
    data = json.loads(config_path.read_text())
    if data["schema"] != 1 or data["profile"] != profile:
        raise ValueError("Unsupported hand collision proxy schema/profile")
    return data


def for_link(config, side, link_name):
    """Return one geom or a list of convex components, in the URDF frame."""
    entry = config["hands"][side][link_name]
    if entry['type'] == 'compound':
        parts = entry.get('parts', [])
        if not parts:
            raise ValueError(f'Empty compound collider: {side}/{link_name}')
        return [_mesh_attributes(part['mesh_file']) for part in parts]
    if entry["type"] == "mesh":
        mesh_file = entry.get("mesh_file")
        if not mesh_file or entry.get("preserve_original_mesh"):
            return None
        return _mesh_attributes(mesh_file)
    allowed = {"type", "pos", "quat", "size", "fromto"}
    attributes = {key: value for key, value in entry.items() if key in allowed}
    if attributes["type"] not in ("capsule", "sphere", "cylinder", "box"):
        raise ValueError(f"Invalid primitive for {link_name}")
    for key in allowed - {"type"}:
        if key in attributes and not np.isfinite(attributes[key]).all():
            raise ValueError(f"Nonfinite proxy {link_name}/{key}")
    if np.any(np.asarray(attributes["size"]) <= 0):
        raise ValueError(f"Nonpositive proxy size for {link_name}")
    return attributes


def _mesh_attributes(mesh_file):
    assets = CONFIG_PATH.parent.resolve()
    path = (assets / mesh_file).resolve()
    if not path.is_relative_to(assets) or not path.is_file():
        raise ValueError(f"Collision mesh escapes/missing from asset root: {mesh_file!r}")
    return {"type": "mesh", "_mesh_file": str(path)}
