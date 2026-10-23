"""Current all-finger targets: white 40 mm / black 20 mm from +world-X end."""
from .forearm_collision import collision_scope

DISTANCE_FROM_POSITIVE_X_END_M = {"white": 0.040, "black": 0.020}
DEFAULT_Z_OFFSET_HALF_SIZE = 0.5
THUMB_GLOBAL_INDICES = (0, 5)


def key_kind(geom_name):
    for colour in ("white", "black"):
        if colour + "_key_geom" in geom_name:
            return colour
    raise ValueError(f"Cannot classify piano key geom: {geom_name}")


def target_position(geom_position, geom_size, geom_name, *, thumb=False,
                    task_profile='tsensei_v1'):
    """World-axis offset for every finger; thumb is an identity label only."""
    import numpy as np
    collision_scope(task_profile)
    position = np.asarray(geom_position, dtype=float).copy()
    size = np.asarray(geom_size, dtype=float)
    if (position.shape != (3,) or size.shape != (3,) or
            not np.isfinite(position).all() or not np.isfinite(size).all() or (size <= 0).any()):
        raise ValueError("Expected finite positions and positive three-dimensional half-sizes")
    distance = DISTANCE_FROM_POSITIVE_X_END_M[key_kind(geom_name)]
    if distance > 2 * size[0]:
        raise ValueError('Target distance exceeds the key length')
    position[0] += size[0] - distance
    position[2] += DEFAULT_Z_OFFSET_HALF_SIZE * size[2]
    return position


def manifest_config(task_profile='tsensei_v1'):
    collision_scope(task_profile)
    return dict(schema=4, geometry_reference="key_body_upper_surface_site_v1", coordinate_frame='world_axes', reference_edge='positive_world_x_end',
                distance_toward_negative_world_x_m=dict(DISTANCE_FROM_POSITIVE_X_END_M),
                z_offset_half_size=DEFAULT_Z_OFFSET_HALF_SIZE, fingers='all_ten')


def options_from_manifest(manifest, task_profile='tsensei_v1'):
    """Reject old geometry, missing contracts, and modified target distances."""
    if manifest != manifest_config(task_profile):
        raise ValueError('Unsupported fingering target geometry; requires white 40 mm / black 20 mm')
    return None


def key_reference(model, geom_id):
    """Return the key's body-centered reference site and upper-face half-sizes.

    Mesh geom_size/geom_xpos refer to principal axes and the mesh centroid;
    they cannot describe the playing surface or the user-preserved key center.
    """
    name = model.geom(int(geom_id)).name
    key_kind(name)
    site = model.site(name.replace("_key_geom_", "_key_site_"))
    if site.bodyid != model.geom(int(geom_id)).bodyid:
        raise ValueError("Key reference must belong to the physical key body")
    return site.id, site.size.copy()


def target_for_key(model, data, geom_id, **kwargs):
    site_id, size = key_reference(model, geom_id)
    return target_position(data.site_xpos[site_id], size, model.geom(int(geom_id)).name, **kwargs)
