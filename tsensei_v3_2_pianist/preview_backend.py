"""Select graphics drivers before importing MuJoCo, GLFW or PyOpenGL."""
import os
from pathlib import Path
import sys


def configure_backend(*, headless=False, gpu='auto'):
    if gpu not in ('auto', 'nvidia', 'system'):
        raise ValueError(f'Unknown GPU selection: {gpu}')
    nvidia_present = sys.platform.startswith('linux') and Path('/proc/driver/nvidia/version').exists()
    if gpu == 'nvidia' or (gpu == 'auto' and nvidia_present):
        for key, value in {'__NV_PRIME_RENDER_OFFLOAD': '1',
                           '__GLX_VENDOR_LIBRARY_NAME': 'nvidia'}.items():
            if gpu == 'nvidia':
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)
    os.environ.setdefault('MUJOCO_GL', 'egl' if headless else 'glfw')


def current_gl_info():
    from OpenGL import GL
    result = {}
    for key, token in [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]:
        value = GL.glGetString(token)
        result[key] = value.decode() if value else 'unknown'
    result['software_renderer'] = any(term in result['renderer'].lower()
                                      for term in ('llvmpipe', 'softpipe', 'software', 'swrast'))
    return result


def probe_desktop_renderer():
    """Small hidden context using the same GLFW settings as the passive viewer."""
    import glfw
    if not glfw.init():
        raise RuntimeError('Cannot open desktop display. Use --headless for an offscreen image.')
    window = None
    try:
        glfw.window_hint(glfw.VISIBLE, False)
        window = glfw.create_window(32, 32, 'Preview GPU probe', None, None)
        if not window:
            raise RuntimeError('Cannot create OpenGL context; try --gpu system or --headless.')
        glfw.make_context_current(window)
        return current_gl_info()
    finally:
        if window:
            glfw.destroy_window(window)
        # Restore VISIBLE: the actual preview window must not inherit the probe hint.
        glfw.default_window_hints()
        glfw.terminate()
