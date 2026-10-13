"""Checkpoint progress may vary; source/task/training configuration may not."""
import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/eval_from_source.py'
SPEC = importlib.util.spec_from_file_location('source_replay_config', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def config():
    return {
        'source_runtime': {'source_sha256': {'model.py': 'original'},
                           'preparation': {'mode': 'policy_v1', 'duration': 1.}},
        'scene_sha256': 'scene-original', 'task_sha256': 'task-original',
        'sac': {'gamma': .8}, 'reward_config': {'self_collision': False},
        'evaluation_selection': {'metric': 'music_only_f1_min', 'target': .99},
    }


def snapshot(config, *, step=20, evaluation=None):
    return {**copy.deepcopy(config), 'vector_step': step, 'updates': 128,
            'evaluation_selection_state': {'best': {'vector_step': step}, 'consecutive': 0},
            'last_evaluation': evaluation, 'last_evaluation_vector_step': step if evaluation else None}


@pytest.mark.parametrize('kind,step,evaluation', [
    ('best', 0, {'music_only_f1_min': .97}),
    ('periodic_before_first_evaluation', 10, None),
    ('periodic', 20, {'music_only_f1_min': .95}),
    ('final', 240, {'music_only_f1_min': .96}),
])
def test_snapshot_progress_does_not_require_config_entries(config, kind, step, evaluation):
    MODULE.validate_config(config, snapshot(config, step=step, evaluation=evaluation))


def test_older_checkpoint_without_selection_fields_remains_valid(config):
    MODULE.validate_config(config, {**config, 'vector_step': 1, 'updates': 0})


def test_unknown_metadata_field_still_requires_config(config):
    metadata = snapshot(config)
    metadata['unreviewed_new_setting'] = True
    with pytest.raises(ValueError, match='Incomplete config.json: unreviewed_new_setting'):
        MODULE.validate_config(config, metadata)


@pytest.mark.parametrize('field', ['source_runtime', 'scene_sha256', 'task_sha256',
                                   'sac', 'reward_config', 'evaluation_selection'])
def test_runtime_fields_do_not_hide_configuration_drift(config, field):
    metadata = snapshot(config)
    changed = copy.deepcopy(config)
    changed[field] = {'incorrect': True}
    with pytest.raises(ValueError, match='differs from checkpoint metadata: '+field):
        MODULE.validate_config(changed, metadata)


@pytest.mark.parametrize('field', ['scene_sha256', 'task_sha256', 'sac',
                                   'reward_config', 'evaluation_selection'])
def test_runtime_fields_do_not_hide_incomplete_configuration(config, field):
    metadata = snapshot(config)
    del config[field]
    with pytest.raises(ValueError, match='Incomplete config.json: '+field):
        MODULE.validate_config(config, metadata)


def test_config_without_source_recipe_is_rejected(config):
    metadata = snapshot(config)
    del config['source_runtime']
    with pytest.raises(ValueError, match='requires a source-based config.json'):
        MODULE.validate_config(config, metadata)


def test_even_allowlisted_fields_must_match_if_present_in_config(config):
    metadata = snapshot(config)
    config['last_evaluation_vector_step'] = 999
    with pytest.raises(ValueError, match='differs from checkpoint metadata: last_evaluation_vector_step'):
        MODULE.validate_config(config, metadata)


def test_unknown_extra_config_field_is_rejected(config):
    metadata = snapshot(config)
    config['unreviewed_config_field'] = True
    with pytest.raises(ValueError, match='differs from checkpoint metadata: unreviewed_config_field'):
        MODULE.validate_config(config, metadata)


@pytest.mark.parametrize('step', [0, 100, 210])
def test_pipeline_progress_is_checkpoint_runtime_metadata(config, step):
    config['execution_layout'] = dict(mode='one_vector_pipeline_v1',
                                    collector_device='cuda:0', learner_device='cuda:1',
                                    max_actor_lag_vector_batches=1)
    metadata = snapshot(config, step=step)
    metadata['collector_pipeline_state'] = dict(collected_vector_steps=step,
                                              learner_updated_through_vector_step=step,
                                              pending_vector_step=None)
    MODULE.validate_config(config, metadata)


@pytest.mark.parametrize('change', ['modified', 'missing'])
def test_pipeline_runtime_allowlist_does_not_relax_execution_layout(config, change):
    config['execution_layout'] = dict(mode='one_vector_pipeline_v1',
                                    collector_device='cuda:0', learner_device='cuda:1')
    metadata = snapshot(config)
    metadata['collector_pipeline_state'] = {'pending_vector_step': None}
    if change == 'missing':
        del config['execution_layout']
        message = 'Incomplete config.json: execution_layout'
    else:
        config['execution_layout']['learner_device'] = 'cuda:0'
        message = 'differs from checkpoint metadata: execution_layout'
    with pytest.raises(ValueError, match=message):
        MODULE.validate_config(config, metadata)
