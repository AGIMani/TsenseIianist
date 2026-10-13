"""Read-only follower-angle diagnostics, separate from rewards and selection."""
import torch


class CouplingMetrics:
    value_keys = ('coupling_target_angle_deg', 'coupling_target_error_deg',
                  'coupling_actual_angle_deg', 'coupling_actual_error_deg',
                  'coupling_infeasible', 'coupling_follower_at_limit')
    labels = ('target_angle_deg', 'target_error_deg', 'actual_angle_deg',
              'actual_error_deg', 'infeasible', 'follower_at_limit')
    scopes = ('', 'active_')

    def __init__(self, device, num_envs, enabled):
        self.enabled = enabled
        self.frames = torch.zeros(num_envs, device=device)
        self.valid = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.counts = torch.zeros((num_envs, 2), device=device)
        self.sums = torch.zeros((num_envs, 2, 6), device=device)
        self.error_maxima = torch.zeros((num_envs, 2, 2), device=device)

    def sample(self, info):
        if not self.enabled or not all(key in info for key in (*self.value_keys, 'coupling_active')):
            return
        values = torch.stack([info[key].float() for key in self.value_keys], dim=-1)
        active = info['coupling_active'].bool()
        self.frames += 1
        errors = torch.stack((values[:, :, 1], values[:, :, 3]), dim=-1)
        self.valid &= (torch.isfinite(values).all(dim=(1, 2))
                       & (errors >= 0).all(dim=(1, 2))
                       & ((values[:, :, 4:] == 0) | (values[:, :, 4:] == 1)).all(dim=(1, 2)))
        weights = torch.stack((torch.ones_like(active), active), dim=1).float()
        values = torch.nan_to_num(values)
        self.counts += weights.sum(-1)
        self.sums += (values[:, None, :, :] * weights[:, :, :, None]).sum(2)
        errors = torch.stack((values[:, :, 1], values[:, :, 3]), dim=-1)
        self.error_maxima.copy_(torch.maximum(self.error_maxima,
            (errors[:, None, :, :] * weights[:, :, :, None]).amax(2)))

    def result(self, length, world):
        if not self.enabled:
            return {'finger_coupling_enabled': False}
        covered = self.frames[world].item() == length and bool(self.valid[world].item())
        result = dict(finger_coupling_enabled=True,
                      finger_coupling_diagnostics_coverage=self.frames[world].item()/max(length, 1),
                      finger_coupling_diagnostics_valid=covered)
        counts = self.counts[world].cpu().tolist()
        sums = self.sums[world].cpu().tolist()
        maxima = self.error_maxima[world].cpu().tolist()
        for scope, count, totals, maximum in zip(self.scopes, counts, sums, maxima):
            prefix = 'finger_coupling_' + scope
            result[prefix+'samples'] = count if covered else None
            for label, total in zip(self.labels, totals):
                suffix = '_fraction' if label in self.labels[4:] else '_mean'
                result[prefix+label+suffix] = total/count if covered and count else None
            for label, value in zip(('target_error_deg', 'actual_error_deg'), maximum):
                result[prefix+label+'_max'] = value if covered and count else None
        return result

    @classmethod
    def aggregate(cls, rows):
        enabled = any(row['finger_coupling_enabled'] for row in rows)
        if not enabled:
            return {'finger_coupling_enabled': False}
        valid = all(row.get('finger_coupling_diagnostics_valid', False) for row in rows)
        result = dict(finger_coupling_enabled=True, finger_coupling_diagnostics_valid=valid,
                      finger_coupling_diagnostics_coverage=min(
                          row.get('finger_coupling_diagnostics_coverage', 0.) for row in rows))
        for scope in cls.scopes:
            prefix = 'finger_coupling_' + scope
            count_key = prefix+'samples'
            count = sum(row[count_key] for row in rows) if valid else None
            result[count_key] = count
            present = [row for row in rows if valid and row[count_key]]
            for label in cls.labels:
                suffix = '_fraction' if label in cls.labels[4:] else '_mean'
                key = prefix+label+suffix
                result[key] = sum(row[key]*row[count_key] for row in present)/count if count else None
            for label in ('target_error_deg', 'actual_error_deg'):
                key = prefix+label+'_max'
                result[key] = max(row[key] for row in present) if present else None
        return result
