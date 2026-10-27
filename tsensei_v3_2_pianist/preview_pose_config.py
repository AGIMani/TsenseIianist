# 2026-10-25: Virtual palm_base naming unified; axis values and order unchanged.
"""Save preview joint targets/positions into the standalone hand constants file."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pprint
import tempfile

import numpy as np


def _replace_values(source, updates):
    """Replace only named assignment values; keep unrelated code and comments."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    def offset(line, byte_column):
        return sum(map(len, lines[:line - 1])) + len(
            lines[line - 1].encode('utf-8')[:byte_column].decode('utf-8'))
    replacements = []
    for name, value in updates.items():
        matches = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == name]
        if len(matches) != 1:
            raise ValueError(f'Expected one top-level assignment for {name}; file was not changed')
        node = matches[0].value
        replacements.append((offset(node.lineno, node.col_offset),
                             offset(node.end_lineno, node.end_col_offset),
                             pprint.pformat(value, width=88, sort_dicts=False)))
    for start, end, text in sorted(replacements, reverse=True):
        source = source[:start] + text + source[end:]
    compile(source, '<saved hand constants>', 'exec')  # Validate without executing.
    return source


class InitialPoseSaver:
    """Back up and atomically update constants; never modify the running model.

    ctrl saves Control targets (including when paused); qpos saves the actual
    visible joints. Both save the mounting position separately from palm_base DOFs.
    """

    def __init__(self, physics, hands, constants_path, backup_dir, source='ctrl'):
        if source not in ('ctrl', 'qpos'):
            raise ValueError('Pose source must be ctrl or qpos')
        self.physics, self.hands, self.source = physics, tuple(hands), source
        self.path, self.backup_dir = Path(constants_path).resolve(), Path(backup_dir)
        self._expected = self.path.read_bytes()

    def _capture(self):
        updates, sides, observations = {}, {}, {}
        for hand in self.hands:
            side = hand.hand_side.name.lower()
            body_id = self.physics.model.ptr.body(hand.root_body.full_identifier).id
            position = self.physics.model.ptr.body_pos[body_id]
            if not np.isfinite(position).all():
                raise ValueError(f'{side} mounting position is not finite')
            updates[side.upper() + '_POSITION'] = tuple(float(v) for v in position)
            values, actual, targets = {}, {}, {}
            for joint in hand.joints:
                actuator = hand.actuator_for_joint(joint.name)
                if actuator is None:
                    raise ValueError(f'Cannot save unactuated joint {joint.name}')
                short = joint.name.removeprefix(side + '_hand_').removesuffix('_joint')
                qpos = float(np.asarray(self.physics.bind(joint).qpos).item())
                ctrl = float(np.asarray(self.physics.bind(actuator).ctrl).item())
                value = ctrl if self.source == 'ctrl' else qpos
                lower = max(float(joint.range[0]), float(actuator.ctrlrange[0]))
                upper = min(float(joint.range[1]), float(actuator.ctrlrange[1]))
                if not np.isfinite(value) or not lower <= value <= upper:
                    raise ValueError(f'{side}/{short}: {self.source}={value} outside '
                                     f'[{lower}, {upper}]; no file written. '
                                     'Use ctrl to save valid targets when actual joints exceed limits.')
                values[short], actual[short], targets[short] = value, qpos, ctrl
            sides[side] = values
            observations[side] = {
                'actual_qpos': actual, 'control_targets': targets,
                'observed_palm_base_world_position_m': self.physics.data.ptr.xpos[body_id].tolist(),
                'observed_palm_base_world_quaternion_wxyz': self.physics.data.ptr.xquat[body_id].tolist(),
                'mount_quaternion_wxyz': self.physics.model.ptr.body_quat[body_id].tolist(),
            }
        if set(sides) != {'left', 'right'}:
            raise ValueError('Saving initial pose requires both hands')
        updates['INITIAL_ANGLES_BY_SIDE'] = sides
        return updates, observations

    def save(self):
        if self.path.read_bytes() != self._expected:
            raise ValueError('Constants changed since preview started; restart preview before saving')
        updates, observations = self._capture()
        updated = _replace_values(self._expected.decode('utf-8'), updates).encode('utf-8')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = self.backup_dir / (stamp + '_' + self.path.name)
        backup.write_bytes(self._expected)
        record = {'time_utc': stamp, 'source': self.source, 'constants_file': str(self.path),
                  'backup_file': str(backup), 'saved_values': updates, 'observations': observations,
                  'before_sha256': hashlib.sha256(self._expected).hexdigest(),
                  'after_sha256': hashlib.sha256(updated).hexdigest(),
                  'references': ['preview_pose_config.py', 'models/hands/tsensei_hand.py',
                                 'models/hands/tsensei_hand_constants.py'],
                  'validation': 'finite values, joint/control limits, Python syntax; no training validation'}
        # Keep a record beside the backup even if a subsequent write fails.
        record_path = backup.with_suffix('.json')
        record['status'] = 'prepared'
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix='.initial_pose_',
                                             delete=False) as output:
                temporary = Path(output.name)
                output.write(updated)
                output.flush()
                os.fsync(output.fileno())
            temporary.chmod(self.path.stat().st_mode & 0o777)
            if self.path.read_bytes() != self._expected:
                raise ValueError('Constants changed while saving; file was not overwritten')
            os.replace(temporary, self.path)
            self._expected = updated
        except Exception as error:
            record.update(status='failed', error=str(error))
            record_path.write_text(json.dumps(record, indent=2) + '\n')
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        record['status'] = 'saved'
        try:
            record_path.write_text(json.dumps(record, indent=2) + '\n')
        except OSError as error:
            print(f'Pose saved, but backup record status could not be updated: {error}', flush=True)
        print(f'Saved initial pose ({self.source}): {self.path}\nBackup: {backup}\n'
              'Restart preview/training to load it. R still restores this window\'s entry pose.', flush=True)
        return record
