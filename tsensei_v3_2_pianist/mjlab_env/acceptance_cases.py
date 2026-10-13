"""Deterministic, physical control cases shared by export and GPU acceptance."""
def press_action(bundle, sides=('R', 'L')):
    """Press within the 25 mm wrist range, with 0.1 rad extra middle flexion."""
    names = bundle.manifest['action_names']
    action = bundle.neutral_action.copy()
    for side in sides:
        wrist = names.index(side+'_Wrist_TZ')
        finger = names.index(side+'_Mid_MCP')
        action[wrist] = max(-.025, bundle.low[wrist])
        action[finger] = min(action[finger]+.1, bundle.high[finger])
    return action


def scenarios(bundle):
    import numpy as np
    names = bundle.manifest['action_names']
    neutral = bundle.neutral_action.copy()
    def target(changes):
        a = neutral.copy()
        for name, value in changes.items():
            a[names.index(name)] = value
        return a
    yield 'neutral', np.tile(neutral, (10, 1))
    for side in ('R', 'L'):
        name = f'{side}_Index_MCP'
        idx = names.index(name)
        yield f'{side}_finger', np.tile(target({name: min(neutral[idx]+.15, bundle.high[idx])}), (10, 1))
    for suffix, value in [('TX', .012), ('TY', .008), ('TZ', -.005), ('Pitch', .10), ('Yaw', .15)]:
        yield f'wrist_{suffix}', np.tile(target({f'{s}_Wrist_{suffix}': value for s in ('R', 'L')}), (10, 1))
    # Actual hand actuation against unmodified piano keys: no forced key qpos.
    press = press_action(bundle)
    yield 'press_release', np.concatenate((np.tile(press, (12, 1)), np.tile(neutral, (20, 1))))
    yield 'sustain', np.tile(target({'sustain': 1.0}), (2, 1))
