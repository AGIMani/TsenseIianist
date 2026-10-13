"""Driver choice must work before any graphics libraries are imported."""
import os
import pytest
from tsensei_v3_2_pianist import preview_backend

KEYS = ('__NV_PRIME_RENDER_OFFLOAD', '__GLX_VENDOR_LIBRARY_NAME', 'MUJOCO_GL')

@pytest.mark.parametrize('gpu,present,expected', [
    ('auto', True, 'nvidia'), ('auto', False, None),
    ('nvidia', False, 'nvidia'), ('system', True, None),
])
def test_gpu_selection(monkeypatch, gpu, present, expected):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(preview_backend.sys, 'platform', 'linux')
    monkeypatch.setattr(preview_backend.Path, 'exists', lambda self: present)
    preview_backend.configure_backend(gpu=gpu)
    assert os.environ.get('__GLX_VENDOR_LIBRARY_NAME') == expected
    assert os.environ.get('__NV_PRIME_RENDER_OFFLOAD') == ('1' if expected else None)
    assert os.environ['MUJOCO_GL'] == 'glfw'


def test_auto_preserves_explicit_driver_and_headless_backend(monkeypatch):
    for key, value in zip(KEYS, ('0', 'mesa', 'osmesa')):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(preview_backend.sys, 'platform', 'linux')
    monkeypatch.setattr(preview_backend.Path, 'exists', lambda self: True)
    preview_backend.configure_backend(headless=True)
    assert tuple(os.environ[key] for key in KEYS) == ('0', 'mesa', 'osmesa')
    preview_backend.configure_backend(gpu='nvidia')
    assert tuple(os.environ[key] for key in KEYS) == ('1', 'nvidia', 'osmesa')
    monkeypatch.delenv('MUJOCO_GL')
    preview_backend.configure_backend(headless=True, gpu='system')
    assert os.environ['MUJOCO_GL'] == 'egl'
