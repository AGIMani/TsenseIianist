"""Shared review camera; display settings never change the physical scene."""
import math
def configure_camera(camera, model, *, view='front', framing='full', distance=None, elevation=None):
    if view not in ('front', 'performer', 'overview'):
        raise ValueError(f'Unknown camera view: {view}')
    if framing not in ('full', 'hands'):
        raise ValueError(f'Unknown framing: {framing}')
    front = view != 'overview'
    camera.orthographic = int(front)
    # Free-camera rendering reads the model globals, not only MjvCamera flags.
    model.vis.global_.orthographic = int(front)
    camera.lookat[:] = (.08, 0, .07) if front and framing == 'hands' else (.03, 0, .06)
    camera.azimuth = {'front': 0, 'performer': 180, 'overview': 155}[view]
    camera.distance = distance if distance is not None else (.55 if front and framing == 'hands' else 1.05)
    # A lower frontal view exposes the vertical gap between fingers and keys.
    camera.elevation = elevation if elevation is not None else {'front': -30, 'performer': -60, 'overview': -50}[view]
    model.vis.global_.fovy = 2*camera.distance*math.tan(math.radians(22.5)) if front else 45
    return dict(view=view, framing=framing, orthographic=bool(camera.orthographic), lookat=camera.lookat.tolist(),
                azimuth=camera.azimuth, elevation=camera.elevation, distance=camera.distance, fovy=float(model.vis.global_.fovy),
                fovy_units='meters (orthographic height)' if front else 'degrees')
