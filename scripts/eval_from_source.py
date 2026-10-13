#!/usr/bin/env python3
"""Replay from current source + complete config.json + a Torch checkpoint.

Use ./eval.sh --checkpoint RUN/final.pt --mode preview
or ./eval.sh --checkpoint RUN/final.pt --mode render --output preview/outputs/NAME
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Training settings live in config.json. These explicitly named snapshot fields
# evolve during a run and are saved only in checkpoint metadata. Any other
# metadata field still requires an equal config value, including future fields.
CHECKPOINT_RUNTIME_METADATA_KEYS = frozenset({
    'vector_step', 'updates',
    'evaluation_selection_state', 'last_evaluation', 'last_evaluation_vector_step',
    'collector_pipeline_state',
})


def validate_config(config, metadata):
    if 'source_runtime' not in config:
        raise ValueError('This entrypoint requires a source-based config.json')
    for key, value in config.items():
        if key not in metadata or json.dumps(value, sort_keys=True) != json.dumps(metadata[key], sort_keys=True):
            raise ValueError('config.json differs from checkpoint metadata: '+key)
    required = set(metadata) - CHECKPOINT_RUNTIME_METADATA_KEYS
    if not required.issubset(config):
        raise ValueError('Incomplete config.json: '+', '.join(sorted(required-set(config))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, help='Defaults to config.json beside checkpoint')
    p.add_argument('--mode', choices=('evaluate', 'preview', 'render'), default='evaluate')
    a, rest = p.parse_known_args()
    if any(s.split('=')[0] in ('--scene', '--task') for s in rest):
        p.error('Replay reconstructs the environment from source and config.json')
    import torch
    metadata = torch.load(a.checkpoint, map_location='cpu', weights_only=True)['metadata']
    config = json.loads((a.config or a.checkpoint.parent/'config.json').read_text())
    validate_config(config, metadata)
    sys.argv = [sys.argv[0], '--checkpoint', str(a.checkpoint), *rest]
    if a.mode == 'evaluate':
        runpy.run_module('tsensei_v3_2_pianist.rl_torch.evaluate', run_name='__main__')
    else:
        name = 'preview_mjlab.py' if a.mode == 'preview' else 'render_mjlab_checkpoint.py'
        runpy.run_path(str(ROOT/'preview/scripts'/name), run_name='__main__')


if __name__ == '__main__':
    main()
