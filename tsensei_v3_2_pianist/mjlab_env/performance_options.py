"""Current options. Key and fingering rewards uses physical samples and original MIDI events.

Other rewards retain their own existing sampling contracts. Historical
sampling switches are removed; correct keys use continuous travel and wrong
keys use actual activation at every physical sample.
"""
import math


DEFAULTS = dict(anticipatory_fingering_time=0., exact_note_timing=True,
                upcoming_press_permission_time=0.)
REMOVED_SAMPLING_OPTIONS = frozenset((
    'substep_rewards', 'target_key_substep_reward',
    'target_key_activation_weight', 'reward_sampling',
    'staged_fingering_reward', 'anticipatory_current_priority',
    'anticipatory_fingering_reward'))


def canonical(options=None):
    if options is None:
        return DEFAULTS.copy()
    if isinstance(options, dict) and REMOVED_SAMPLING_OPTIONS.intersection(options):
        raise ValueError('Removed reward sampling options: '+', '.join(sorted(
            REMOVED_SAMPLING_OPTIONS.intersection(options)))+
            '; use the matching archived source for historical configurations')
    if not isinstance(options, dict) or not set(options) <= set(DEFAULTS):
        raise ValueError('performance accepts only '+', '.join(DEFAULTS))
    result = dict(DEFAULTS, **options)
    horizon = result['anticipatory_fingering_time']
    if (isinstance(horizon, bool) or not isinstance(horizon, (int, float))
            or not math.isfinite(horizon) or not 0. <= horizon <= .5):
        raise ValueError('anticipatory_fingering_time must be finite seconds in [0, 0.5]')
    result['anticipatory_fingering_time'] = float(horizon)
    if type(result['exact_note_timing']) is not bool or not result['exact_note_timing']:
        raise ValueError('Current rewards require exact_note_timing=true')
    window = result['upcoming_press_permission_time']
    if (isinstance(window, bool) or not isinstance(window, (int, float))
            or not math.isfinite(window) or not 0. <= window <= .15):
        raise ValueError('upcoming_press_permission_time must be finite seconds in [0, 0.15]')
    result['upcoming_press_permission_time'] = float(window)
    if window and (not result['exact_note_timing'] or window > horizon):
        raise ValueError('Upcoming press permission requires exact_note_timing and anticipation >= permission time')
    return result


def get(reward_config):
    return dict(DEFAULTS, **canonical(reward_config.get('performance')))


def active(reward_config):
    return any(value != DEFAULTS[name] for name, value in get(reward_config).items())
