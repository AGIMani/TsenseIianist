"""Audio and task observers must see the identical complete physical replay."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[1] / 'preview/scripts/render_mjlab_checkpoint.py'
spec = importlib.util.spec_from_file_location('render_audio_callback_test', path)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def fake_physics(log):
    state = SimpleNamespace(substep=0)
    def step(action, *, substep_callback, assigned_keys):
        assert action == 'policy action'
        assert assigned_keys == 'original fingering'
        for _ in range(10):
            state.substep += 1
            log.append(('physics', state.substep))
            substep_callback(state)
        return state.substep
    return step


def test_task_and_audio_callbacks_each_observe_every_substep_once():
    log = []
    step = renderer._with_substep_observer(fake_physics(log),
        lambda state: log.append(('audio', state.substep)))
    task = lambda state: log.append(('tracking', state.substep))
    for expected in (10, 20):
        assert step('policy action', assigned_keys='original fingering',
                    substep_callback=task) == expected
    assert log == [(kind, tick) for tick in range(1, 21)
                   for kind in ('physics', 'tracking', 'audio')]


@pytest.mark.parametrize('kwargs', [{}, {'substep_callback': None}])
def test_audio_recording_survives_absent_or_explicit_none_task_callback(kwargs):
    log = []
    step = renderer._with_substep_observer(fake_physics(log),
        lambda state: log.append(('audio', state.substep)))
    assert step('policy action', assigned_keys='original fingering', **kwargs) == 10
    assert log == [(kind, tick) for tick in range(1, 11) for kind in ('physics', 'audio')]


def test_audio_composes_with_an_existing_archive_trace_observer():
    log = []
    # The archive recorder wraps the scene step before the renderer installs
    # audio recording; both must retain the task's tracking callback.
    trace_step = renderer._with_substep_observer(fake_physics(log),
        lambda state: log.append(('trace', state.substep)))
    step = renderer._with_substep_observer(trace_step,
        lambda state: log.append(('audio', state.substep)))
    step('policy action', assigned_keys='original fingering',
         substep_callback=lambda state: log.append(('tracking', state.substep)))
    assert log == [(kind, tick) for tick in range(1, 11)
                   for kind in ('physics', 'tracking', 'audio', 'trace')]
