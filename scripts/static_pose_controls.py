"""Mouse controls for a display-only joint pose; no simulation or saved edits."""
import math


class PoseControls:
    width = 370
    groups = ('Wrist', 'Thumb', 'Index', 'Mid', 'Ring', 'Pinky')

    def __init__(self, model, data, refresh):
        self.model, self.data, self.refresh = model, data, refresh
        self.initial = data.qpos.copy()
        self.side, self.group = 'right', 'Wrist'
        self.controls, self.buttons, self.sliders = [], [], []
        self.drag = None
        self.edits = 0
        for j in range(model.njnt):
            name = model.joint(j).name or ''
            if not name.startswith(('right_/', 'left_/')):
                continue
            # MuJoCo scalar slide/hinge joints only; free/ball joints are excluded.
            if int(model.jnt_type[j]) not in (2, 3) or not model.jnt_limited[j]:
                continue
            side, short = name.split('_/', 1)
            short = short.removeprefix(side + '_hand_').removesuffix('_joint')
            group = 'Wrist' if short.startswith('palm_base_') else short.split('_')[0].title()
            if group not in self.groups:
                continue
            low, high = map(float, model.jnt_range[j])
            if not math.isfinite(low + high) or high <= low:
                continue
            self.controls.append(dict(name=name, label=short, side=side, group=group,
                address=int(model.jnt_qposadr[j]), low=low, high=high,
                unit='m' if int(model.jnt_type[j]) == 2 else 'rad'))

    @property
    def addresses(self):
        return [c['address'] for c in self.controls]

    @staticmethod
    def contains(rect, x, y):
        left, top, width, height = rect
        return left <= x <= left + width and top <= y <= top + height

    def set_from_pointer(self, rect, control, x):
        if not math.isfinite(x):
            return
        fraction = max(0., min(1., (x - rect[0]) / rect[2]))
        self.data.qpos[control['address']] = control['low'] + fraction * (control['high'] - control['low'])
        self.edits += 1
        self.refresh()

    def reset(self):
        self.data.qpos[self.addresses] = self.initial[self.addresses]
        self.edits += 1
        self.refresh()

    def press(self, x, y, window_width):
        for rect, kind, value in self.buttons:
            if self.contains(rect, x, y):
                if kind == 'reset':
                    self.reset()
                else:
                    setattr(self, kind, value)
                self.drag = None
                return True
        for rect, control in self.sliders:
            if self.contains(rect, x, y):
                self.drag = (rect, control)
                self.set_from_pointer(rect, control, x)
                return True
        return x >= window_width - self.width

    def move(self, x):
        if self.drag is None:
            return False
        self.set_from_pointer(*self.drag, x)
        return True

    def release(self):
        self.drag = None

    def draw(self, window, context):
        import glfw
        import mujoco
        width, height = glfw.get_framebuffer_size(window)
        ww, hh = glfw.get_window_size(window)
        if min(ww, hh, width, height) <= 0:
            return
        sx, sy = width / ww, height / hh

        def rectangle(rect, color):
            x, y, w, h = rect
            mujoco.mjr_rectangle(mujoco.MjrRect(int(x*sx), int((hh-y-h)*sy),
                int(w*sx), int(h*sy)), *color)

        def label(rect, text):
            x, y, w, h = rect
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                mujoco.MjrRect(int(x*sx), int((hh-y-h)*sy), int(w*sx), int(h*sy)), text, '', context)

        rectangle((ww-self.width, 0, self.width, hh), (.075, .09, .12, 1.))
        left = ww-self.width+14
        label((left, 10, 344, 48), 'JOINT POSE | No physics\nDrag sliders to adjust pose')
        self.buttons.clear()
        for i, side in enumerate(('right', 'left')):
            rect = (left+i*170, 64, 160, 30)
            rectangle(rect, (.16, .38, .48, 1.) if self.side == side else (.16, .19, .24, 1.))
            label(rect, side.title()+' hand')
            self.buttons.append((rect, 'side', side))
        for i, group in enumerate(self.groups):
            rect = (left+(i%3)*113, 103+(i//3)*36, 106, 30)
            rectangle(rect, (.16, .38, .48, 1.) if self.group == group else (.16, .19, .24, 1.))
            label(rect, group)
            self.buttons.append((rect, 'group', group))
        self.sliders.clear()
        for i, c in enumerate(c for c in self.controls if c['side'] == self.side and c['group'] == self.group):
            y = 184+i*76
            value = float(self.data.qpos[c['address']])
            label((left, y, 344, 46), f'{c["label"]}: {value:.4f} {c["unit"]}\n[{c["low"]:.4f}, {c["high"]:.4f}]')
            rect = (left+8, y+48, 320, 17)
            rectangle(rect, (.18, .22, .28, 1.))
            ratio = max(0., min(1., (value-c['low'])/(c['high']-c['low'])))
            rectangle((rect[0], rect[1], max(2, rect[2]*ratio), rect[3]), (.25, .66, .82, 1.))
            self.sliders.append(((rect[0], y+44, rect[2], 25), c))
        rect = (left, 510, 336, 32)
        rectangle(rect, (.16, .3, .25, 1.))
        label(rect, 'Reset both hands (R)')
        self.buttons.append((rect, 'reset', None))
        label((left, 555, 344, 125), 'Direct pose / model joint limits\nPIP and DIP move independently.\nNo coupling or contact scoring.\nNo forces or time integration.\nTraining defaults are not edited.')
