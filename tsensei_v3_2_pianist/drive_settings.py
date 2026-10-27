# 2026-10-25: Virtual palm_base naming unified; axis values and order unchanged.
"""Explicit standalone drive profiles; old source behavior stays the default."""
import json
import math
import os
from pathlib import Path


SETTINGS_PATH = Path(__file__).resolve().parent / 'assets' / 'drive_settings.json'
PROFILE_ENV = 'TSENSEI_V32_DRIVE_PROFILE'
PROFILES = ('legacy_v1', 'shadow_v1')


def translation_damping(settings, compiled_inertia, joint):
    """Return the selected axis damping after checking the compiled slide mass.

    This is a scalar, contact-free reference, not a closed-loop tracking or
    hardware guarantee. Force limits are independent of the armature value.
    """
    expected = settings['effective_mass_reference_kg']
    if (not math.isfinite(compiled_inertia) or compiled_inertia <= 0
            or not math.isclose(compiled_inertia, expected, rel_tol=1e-6, abs_tol=1e-9)):
        raise ValueError(f'Translation compiled inertia {compiled_inertia:g} kg '
                         f'does not match drive reference {expected:g} kg; '
                         'review hand mass and translation armature')
    return settings['axes'][joint]['damping']


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
        for name in ('pitch',):
            for key, value in result[name].items():
                if key == 'reference':
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'Invalid drive value {name}.{key}')
        if result['finger'] != {'parameter_file': 'finger_resistance.json', 'contract': 'a3_resistance_compensated_v1'}:
            raise ValueError('Unsupported finger drive contract')
        translation = result['translation']
        if (translation['joints'] != ['palm_base_tx', 'palm_base_ty']
                or not isinstance(translation.get('axes'), dict)
                or set(translation.get('axes', {})) != set(translation['joints'])
                or any(key in translation for key in ('kp', 'damping', 'force_limit'))):
            raise ValueError('Unsupported translation drive contract')
        for joint, axis in translation['axes'].items():
            if (not isinstance(axis, dict) or set(axis) != {'kp', 'damping', 'force_limit'}
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or v <= 0 for v in axis.values())):
                raise ValueError(f'Invalid translation drive value: {joint}')
        if any(isinstance(translation[k], bool)
               or not isinstance(translation[k], (int, float))
               or not math.isfinite(translation[k]) or translation[k] <= 0
               for k in ('armature', 'hand_mass_kg',
                         'effective_mass_reference_kg')):
            raise ValueError('Invalid translation drive value')
        expected_mass = translation['hand_mass_kg'] + translation['armature']
        if not math.isclose(translation['effective_mass_reference_kg'], expected_mass,
                            rel_tol=0., abs_tol=1e-12):
            raise ValueError('Translation effective mass does not match current hand mass plus armature')
    return result
