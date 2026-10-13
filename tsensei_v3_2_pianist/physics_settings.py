"""User-editable contact settings; geometry and actuator calibration stay separate."""
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import os
import tempfile

SETTINGS_PATH = Path(__file__).parent/'assets/contact_settings.json'
GROUPS = ('hand', 'piano', 'key_limit')


def shadow_reference(physics_dt, control_dt, lookahead_seconds=.5):
    """Return RoboPianist Shadow contact values at the active timestep.

    Shadow fixes the hand material and key-limit values below.  The piano
    contact time constant follows RoboPianist's ``2 * physics_dt`` rule, so
    this describes both the original 20/200 Hz model and our 40/400 Hz model
    without changing either model's timing.
    """
    return validate({
        'schema': 1,
        'profile': 'shadow_contact_reference',
        'physics_dt': float(physics_dt),
        'control_dt': float(control_dt),
        'lookahead_seconds': float(lookahead_seconds),
        'integrator': 'implicitfast',
        'impratio': 10.,
        'hand': {'solref': [.005, 1.], 'solimp': [.5, .99, .0001, .5, 2.],
                 'friction': [1., .005, .0001], 'priority': 0},
        'piano': {'solref': [2 * float(physics_dt), 1.],
                  'solimp': [.9, .95, .001, .5, 2.],
                  'friction': [1., .005, .0001], 'priority': 0},
        'key_limit': {'solref': [.02, 1.],
                      'solimp': [.9, .95, .001, .5, 2.]},
    })


def shadow_comparison(settings, atol=1e-12):
    """Compare actual contact values with Shadow; timing itself is unchanged."""
    actual = validate(settings)
    expected = shadow_reference(actual['physics_dt'], actual['control_dt'],
                                actual['lookahead_seconds'])
    differences = {}
    for name in ('integrator', 'impratio'):
        if (actual[name] != expected[name] if name == 'integrator'
                else not math.isclose(actual[name], expected[name], abs_tol=atol, rel_tol=0)):
            differences[name] = {'actual': actual[name], 'shadow': expected[name]}
    for group in GROUPS:
        fields = ('solref', 'solimp') if group == 'key_limit' else (
            'solref', 'solimp', 'friction', 'priority')
        for field in fields:
            a, e = actual[group][field], expected[group][field]
            equal = (a == e) if field == 'priority' else all(
                math.isclose(x, y, abs_tol=atol, rel_tol=0) for x, y in zip(a, e))
            if not equal:
                differences[f'{group}.{field}'] = {'actual': a, 'shadow': e}
    hand, piano = actual['hand'], actual['piano']
    if hand['priority'] == piano['priority']:
        contact_solref = [(a + b) / 2 for a, b in zip(hand['solref'], piano['solref'])]
        contact_solimp = [(a + b) / 2 for a, b in zip(hand['solimp'], piano['solimp'])]
    elif hand['priority'] > piano['priority']:
        contact_solref, contact_solimp = hand['solref'], hand['solimp']
    else:
        contact_solref, contact_solimp = piano['solref'], piano['solimp']
    return {
        'matches_shadow': not differences,
        'differences': differences,
        'hand_key_contact': {
            'mixed_solref': list(contact_solref),
            'refsafe_timeconst_s': max(float(contact_solref[0]), 2 * actual['physics_dt']),
            'mixed_solimp': list(contact_solimp),
            'friction': [max(a, b) for a, b in zip(hand['friction'], piano['friction'])],
        },
        'reference': expected,
    }


def validate(settings):
    s = copy.deepcopy(settings)
    if s['schema'] != 1:
        raise ValueError('Unsupported contact settings schema')
    json.dumps(s, allow_nan=False)
    for key in ('physics_dt', 'control_dt', 'lookahead_seconds', 'impratio'):
        if not isinstance(s[key], (int, float)) or s[key] <= 0:
            raise ValueError(f'{key} must be positive')
    ratio = s['control_dt']/s['physics_dt']
    if not math.isclose(ratio, round(ratio)):
        raise ValueError('Control dt must be an integer multiple of physics dt')
    if s['integrator'] not in ('Euler', 'implicitfast'):
        raise ValueError('Unsupported integrator')
    for name in GROUPS:
        g = s[name]
        if len(g['solimp']) != 5 or len(g['solref']) != 2:
            raise ValueError(f'{name}: solimp requires 5 values; solref requires 2')
        d0, dw, width, mid, power = g['solimp']
        if not (0.0001 <= d0 <= .9999 and .0001 <= dw <= .9999
                and width > 0 and 0 < mid < 1 and power >= 1):
            raise ValueError(f'{name}: invalid solimp')
        if min(g['solref']) <= 0:
            raise ValueError(f'{name}: positive solref required')
        if name != 'key_limit' and (len(g['friction']) != 3 or min(g['friction']) < 0):
            raise ValueError(f'{name}: invalid friction')
    return s


def load(path=SETTINGS_PATH):
    return validate(json.loads(Path(path).read_text()))


def save(settings, path=SETTINGS_PATH, *, expected=None):
    """Explicit save with backup and protection against another editor's changes."""
    path = Path(path)
    raw = path.read_bytes() if path.exists() else None
    if expected is not None and raw != expected:
        raise ValueError('Contact config changed on disk; reopen preview before saving')
    text = json.dumps(validate(settings), indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        backup = path.parent/'contact_settings_backups'
        backup.mkdir(exist_ok=True)
        (backup/f'{path.stem}-{datetime.now():%Y%m%d-%H%M%S-%f}.json').write_bytes(raw)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
            name = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()
    return path.read_bytes()


def apply_mjcf_piano(root, settings):
    for attr in ('solref', 'solimp', 'friction', 'priority'):
        setattr(root.default.geom, attr, settings['piano'][attr])
    root.default.joint.solreflimit = settings['key_limit']['solref']
    root.default.joint.solimplimit = settings['key_limit']['solimp']


def model_indices(model):
    import numpy as np
    hand = [i for i in range(model.ngeom) if (model.geom(i).name or '').startswith(('right_/', 'left_/'))
            and (model.geom_contype[i] or model.geom_conaffinity[i])]
    piano = [i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('piano/')]
    keys = [i for i in range(model.njnt) if (model.joint(i).name or '').startswith('piano/')]
    return {k: np.asarray(v, dtype=int) for k,v in zip(GROUPS, (hand, piano, keys))}


def from_model(model, control_dt):
    s = load()
    s.update(physics_dt=float(model.opt.timestep), control_dt=float(control_dt),
             integrator='Euler' if int(model.opt.integrator) == 0 else 'implicitfast',
             impratio=float(model.opt.impratio))
    for group, ids in model_indices(model).items():
        i = ids[0]
        if group == 'key_limit':
            s[group] = dict(solref=model.jnt_solref[i].tolist(), solimp=model.jnt_solimp[i].tolist())
        else:
            s[group] = {k: getattr(model, 'geom_'+k)[i].tolist()
                        for k in ('solref', 'solimp', 'friction', 'priority')}
    return validate(s)


class SolimpController:
    """Update only impedance arrays in-place, including CUDA graph referenced arrays."""
    def __init__(self, model, settings, scene=None):
        self.model, self.scene = model, scene
        self.settings = validate(settings)
        self.initial = copy.deepcopy(self.settings)
        self.indices = model_indices(model)
        self.dirty = False

    def set_values(self, group, values):
        candidate = copy.deepcopy(self.settings)
        candidate[group]['solimp'] = list(values)
        self.settings = validate(candidate)
        self.dirty = True

    def restore_shadow(self):
        """Restore every slider-controlled impedance to the Shadow reference."""
        reference = shadow_reference(self.settings['physics_dt'], self.settings['control_dt'],
                                     self.settings['lookahead_seconds'])
        for group in GROUPS:
            self.settings[group]['solimp'] = reference[group]['solimp'].copy()
        self.dirty = True

    def apply(self):
        if not self.dirty:
            return
        for group, ids in self.indices.items():
            field = self.model.jnt_solimp if group == 'key_limit' else self.model.geom_solimp
            field[ids] = self.settings[group]['solimp']
        if self.scene is not None:
            import numpy as np
            with self.scene.scope():
                for name in ('geom_solimp', 'jnt_solimp'):
                    dest = getattr(self.scene.sim.wp_model, name)
                    value = getattr(self.model, name)
                    # Preserve array addresses held by captured CUDA graphs.
                    dest.assign(np.broadcast_to(value, dest.numpy().shape).astype(np.float32).copy())
                self.scene.sim.forward()
        self.dirty = False
