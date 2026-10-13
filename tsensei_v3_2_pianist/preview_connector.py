"""Optional connector visuals and editable mounting offsets; no added dynamics."""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/preview/hand_connector_20260930.json'


def offset_transform(offset):
    xyz, rpy = np.asarray(offset['xyz_mm'], dtype=float), np.asarray(offset['rpy_deg'], dtype=float)
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
        raise ValueError('Connector offsets require three finite translations and angles')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
    matrix[:3, 3] = xyz / 1000
    return matrix


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config.get('schema') != 1:
        raise ValueError('Unsupported connector schema')
    for side in ('left', 'right'):
        offset_transform(config['offsets'][side])
        matrix = np.asarray(config['nominal_link7_to_hand'][side], dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-10)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.)):
            raise ValueError('Connector transform must be a rigid proper transform')
    return config


def tool_transforms(config):
    # The adjustment moves the connector/flange in hand coordinates.
    return {side: np.asarray(config['nominal_link7_to_hand'][side]) @
                  np.linalg.inv(offset_transform(config['offsets'][side])) for side in ('left', 'right')}


class ConnectorPreview:
    def __init__(self, root, config_path=DEFAULT_CONFIG):
        self.path = Path(config_path)
        self.expected = self.path.read_bytes()
        self.config = load_config(self.path)
        self.initial = copy.deepcopy(self.config['offsets'])
        self.pending = None
        self.alignment_label = self.config.get('alignment_label', '中央孔已对齐')
        self.status = self.alignment_label + '；平移和旋转使用各自手部坐标。'
        self.bodies = {}
        for side in ('left', 'right'):
            body = root.worldbody.add('body', name=f'hand_connector_{side}', mocap=True)
            self.bodies[side] = body
            for index, name in enumerate(self.config['meshes'][side]):
                path = (ROOT / name).resolve()
                if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != self.config['mesh_sha256'][name]:
                    raise ValueError('Connector mesh differs from saved alignment: ' + name)
                mesh = root.asset.add('mesh', name=f'hand_connector_{side}_{index}', file=str(path))
                body.add('geom', type='mesh', mesh=mesh, contype=0, conaffinity=0, group=1,
                         rgba=[.85, .60, .22, 1.])

    def bind(self, physics, hands, arm_preview=None):
        self.physics, self.arms = physics, arm_preview
        model = physics.model.ptr
        self.hand_ids = {side: model.body(hand.root_body.full_identifier).id
                         for side, hand in zip(('right', 'left'), hands)}
        self.mocap_ids = {side: int(model.body_mocapid[model.body(body.full_identifier).id])
                          for side, body in self.bodies.items()}
        self.publish()

    def request(self, side, xyz_mm, rpy_deg):
        if side not in ('left', 'right'):
            raise ValueError('Connector side must be left or right')
        config = copy.deepcopy(self.pending if self.pending is not None else self.config)
        config['offsets'][side] = {'xyz_mm': list(xyz_mm), 'rpy_deg': list(rpy_deg)}
        offset_transform(config['offsets'][side])
        self.pending = config

    def apply(self):
        if self.pending is None:
            return
        candidate, self.pending = self.pending, None
        try:
            if self.arms is not None:
                self.arms.set_tool_transforms(tool_transforms(candidate))
            self.config = candidate
            self.status = '调整已生效。保存按钮写入安装配置；基准装配网格保持。'
        except ValueError as error:
            self.status = '未应用：' + str(error)

    def publish(self):
        data = self.physics.data.ptr
        for side, bid in self.hand_ids.items():
            matrix = np.eye(4)
            matrix[:3, :3] = data.xmat[bid].reshape(3, 3)
            matrix[:3, 3] = data.xpos[bid]
            matrix = matrix @ offset_transform(self.config['offsets'][side])
            mid = self.mocap_ids[side]
            data.mocap_pos[mid] = matrix[:3, 3]
            data.mocap_quat[mid] = np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)
        self.physics.forward()

    def save(self):
        if self.pending is not None:
            raise ValueError('请等待当前调整生效后再保存')
        if self.path.read_bytes() != self.expected:
            raise ValueError('安装配置已被其他进程修改，请重新打开预览')
        backup = self.path.with_name(self.path.name + '.bak-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        backup.write_bytes(self.expected)
        content = (json.dumps(self.config, ensure_ascii=False, indent=2) + '\n').encode()
        self.path.write_bytes(content)
        self.expected = content
        self.status = '已保存：' + str(self.path)

    def report(self):
        return {'config_path': str(self.path), 'config': self.config,
                'resolved_link7_to_hand': {s: t.tolist() for s, t in tool_transforms(self.config).items()},
                'physical_mass_or_contacts_added': False}


class ConnectorPanel:
    """Companion Tk sliders, polled on the same thread as existing contact UI."""
    def __init__(self, controller):
        import tkinter as tk
        from tkinter import ttk
        self.controller, self.closed, self.loading = controller, False, False
        self.root = tk.Tk()
        self.root.title('连接件安装调整 — ' + controller.alignment_label)
        self.root.geometry('520x650')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.side = tk.StringVar(master=self.root, value='left')
        ttk.Label(self.root, text='连接件相对手部坐标；滑动后双臂重新求解安装姿态。').pack(pady=9)
        selector = ttk.Combobox(self.root, textvariable=self.side, values=('left', 'right'), state='readonly')
        selector.pack()
        selector.bind('<<ComboboxSelected>>', lambda _: self.refresh())
        self.scales = []
        for label, low, high, resolution in [('X 平移 (mm)', -50, 50, .1), ('Y 平移 (mm)', -50, 50, .1),
                                            ('Z 平移 (mm)', -50, 50, .1), ('Roll (deg)', -45, 45, .1),
                                            ('Pitch (deg)', -45, 45, .1), ('Yaw (deg)', -180, 180, .1)]:
            scale = tk.Scale(self.root, label=label, from_=low, to=high, resolution=resolution,
                             orient='horizontal', length=470, command=self.changed)
            scale.pack()
            self.scales.append(scale)
        self.message = tk.StringVar(master=self.root)
        ttk.Label(self.root, textvariable=self.message, wraplength=485).pack(pady=9)
        row = ttk.Frame(self.root)
        row.pack()
        ttk.Button(row, text='恢复基准装配', command=self.reset).pack(side='left', padx=8)
        ttk.Button(row, text='保存安装配置', command=self.save).pack(side='left', padx=8)
        self.refresh()

    def refresh(self):
        self.loading = True
        offset = self.controller.config['offsets'][self.side.get()]
        for scale, value in zip(self.scales, offset['xyz_mm'] + offset['rpy_deg']):
            scale.set(value)
        self.root.update_idletasks()
        self.loading = False

    def changed(self, _=None):
        if not self.loading:
            values = [s.get() for s in self.scales]
            self.controller.request(self.side.get(), values[:3], values[3:])

    def reset(self):
        self.controller.request(self.side.get(), [0.] * 3, [0.] * 3)
        self.loading = True
        for scale in self.scales:
            scale.set(0.)
        self.root.update_idletasks()
        self.loading = False

    def save(self):
        try:
            self.controller.save()
        except (ValueError, OSError) as error:
            self.controller.status = str(error)

    def poll(self):
        if not self.closed:
            self.message.set(self.controller.status)
            self.root.update()

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
