"""Explicit standalone drive profiles; old source behavior stays the default."""
import json
import math
import os
from pathlib import Path


SETTINGS_PATH = Path(__file__).resolve().parent / 'assets' / 'drive_settings.json'
PROFILE_ENV = 'TSENSEI_V32_DRIVE_PROFILE'
PROFILES = ('legacy_v1', 'shadow_v1')


def load(profile=None):
    """Return a fresh profile, rejecting silent fallback on misspelled settings."""
    document = json.loads(SETTINGS_PATH.read_text())
    if document.get('schema') != 1 or document.get('default_profile') != 'legacy_v1':
        raise ValueError('Unsupported drive settings schema/default profile')
    selected = profile if profile is not None else os.environ.get(PROFILE_ENV, 'legacy_v1')
    if selected not in PROFILES or selected not in document['profiles']:
        raise ValueError(f'Unsupported V3.2 drive profile: {selected!r}')
    result = dict(document['profiles'][selected], profile=selected)
    if selected == 'shadow_v1':
        for name in ('finger', 'pitch'):
            for key, value in result[name].items():
                if key == 'reference':
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'Invalid drive value {name}.{key}')
        translation = result['translation']
        if translation['joints'] != ['forearm_tx', 'forearm_ty'] or translation['damping'] != 'critical_from_compiled_dof_M0':
            raise ValueError('Unsupported translation drive contract')
        if any(not math.isfinite(translation[k]) or translation[k] <= 0
               for k in ('kp', 'force_limit', 'armature', 'hand_mass_kg',
                         'effective_mass_reference_kg')):
            raise ValueError('Invalid translation drive value')
        expected_mass = translation['hand_mass_kg'] + translation['armature']
        if not math.isclose(translation['effective_mass_reference_kg'], expected_mass,
                            rel_tol=0., abs_tol=1e-12):
            raise ValueError('Translation effective mass does not match current hand mass plus armature')
    return result
