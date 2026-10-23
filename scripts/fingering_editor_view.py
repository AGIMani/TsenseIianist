"""Display-only camera math: +Y is higher pitch; +Z is up."""
import math
import numpy as np


def pale_color(color):
    """Keep the finger hue, with 30% original color and 70% white."""
    rgb = [int(color[i:i+2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(.30 * value + .70 * 255):02x}' for value in rgb)


def diagonal_hatching(polygon, spacing=9):
    """Clip screen-space / lines to a convex display polygon, without spill."""
    points = np.asarray(polygon, dtype=float)[:, :2]
    sums = points.sum(axis=1)
    start = math.ceil(sums.min()/spacing)*spacing
    for level in np.arange(start, sums.max(), spacing):
        intersections = []
        for i, a in enumerate(points):
            b = points[(i+1) % len(points)]
            delta = b.sum()-a.sum()
            if abs(delta) < 1e-10:
                continue
            t = (level-a.sum())/delta
            if 0 <= t <= 1:
                intersections.append(a+t*(b-a))
        if len(intersections) >= 2:
            intersections.sort(key=lambda p: p[0])
            if np.linalg.norm(intersections[-1]-intersections[0]) > 1e-6:
                yield intersections[0], intersections[-1]


def camera_basis(azimuth, elevation):
    """Screen-right, screen-down, depth-toward-camera for an above-keyboard view.

    Azimuth zero looks from the performer (+X), so right hand (+Y) appears
    on screen right. Positive depth is near: painter sorting must be ascending.
    """
    a, e = azimuth, elevation
    return np.array([
        [math.sin(a), math.cos(a), 0.],
        [math.sin(e)*math.cos(a), -math.sin(e)*math.sin(a), -math.cos(e)],
        [math.cos(e)*math.cos(a), -math.cos(e)*math.sin(a), math.sin(e)],
    ])


def split_key_faces(vertices, faces, colors):
    """Split a multiply assigned key into finger-colored bands along its width.

    Only display polygons are clipped. Original model geometry is unchanged.
    """
    low, high = vertices[:, 1].min(), vertices[:, 1].max()
    edges = np.linspace(low, high, len(colors) + 1)
    for face in faces:
        for i, color in enumerate(colors):
            polygon = vertices[face]
            for bound, sign in ((edges[i], 1), (edges[i + 1], -1)):
                clipped = []
                if len(polygon) < 3:
                    break
                previous = polygon[-1]
                inside_previous = sign * (previous[1] - bound) >= 0
                for point in polygon:
                    inside = sign * (point[1] - bound) >= 0
                    if inside != inside_previous:
                        t = (bound - previous[1]) / (point[1] - previous[1])
                        clipped.append(previous + t * (point - previous))
                    if inside:
                        clipped.append(point)
                    previous, inside_previous = point, inside
                polygon = np.asarray(clipped)
            if len(polygon) >= 3:
                yield polygon, color
