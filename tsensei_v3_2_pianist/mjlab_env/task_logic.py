"""Batched Torch task rules matching the fingered Composer training reference.

No physics stepping and no CPU/NumPy transfers in observation/reward computation.
The lookahead includes the current frame. At score completion the final goal
window is retained, matching Composer's terminal observation.
"""
from dataclasses import dataclass
import math

from ..forearm_collision import forearm_contact_reward
import torch


def gaussian_tolerance(x, upper, margin):
    # dm_control tolerance(bounds=(0, upper), sigmoid='gaussian', value_at_margin=.1)
    distance = torch.maximum(-x, x-upper).clamp_min(0)/margin
    return torch.exp(math.log(.1)*distance.square())


def key_press_substep_reward(snapshot, target, tolerance, *, chord_bonus=False, chord_scope=None):
    """Continuous key-travel reward on exact physical sample timestamps.

    Original event goals may change within a control step. The target fallback
    supports explicit quantized-score tasks/analytic fixtures, never endpoint
    state sampling. Production key reward requires exact score events.
    """
    travel = snapshot['key_state_history']
    activation = snapshot['key_activation_history']
    if travel.ndim != 3 or travel.shape[-1] != 88 or activation.shape != travel.shape:
        raise ValueError('Key reward requires matching [env, physical_substep, 88] histories')
    if travel.shape[1] < 1 or activation.dtype != torch.bool:
        raise ValueError('Key reward requires nonempty physical samples and boolean activation')
    goals = snapshot.get('exact_goal_history', target[:, None, :])
    on = goals[..., :88].bool()
    count = on.sum(-1)
    depth = gaussian_tolerance(1-travel, tolerance, tolerance*10)
    # Restore the continuous correct-key signal without an activation jump.
    # Actual activation still determines every wrong-key penalty and F1.
    positive = (depth*on).sum(-1)/count.clamp_min(1)
    wrong_count = (activation & ~on).sum(-1)
    multiplier = torch.where(count >= 3, 1.15, torch.where(count == 2, 1.08, 1.))
    if chord_bonus and chord_scope == 'on':
        positive = positive*multiplier
    score = .6*positive + .4*torch.exp(-1.25*wrong_count.float())
    if chord_bonus and chord_scope == 'whole_key':
        score = score*multiplier
    return score.mean(-1)


def binary_metrics(goal, actual):
    goal, actual = goal.bool(), actual.bool()
    tp = (goal & actual).sum(-1)
    fp = (~goal & actual).sum(-1)
    fn = (goal & ~actual).sum(-1)
    def ratio(a,b):
        return torch.where(b>0, a.float()/b.clamp_min(1), 1.)
    return torch.stack((ratio(tp,tp+fp),ratio(tp,tp+fn),ratio(2*tp,2*tp+fp+fn)),dim=-1), torch.stack((tp,fp,fn),dim=-1)


def fingering_substep_reward(snapshot, tolerance):
    """Exact-event Gaussian distance mean per substep, then mean over time.

    Idle substeps contribute zero. Future notes never enter this calculation.
    Targets use actual post-integration key geometry, not endpoint geometry.
    """
    tips = snapshot['fingertip_history']
    targets = snapshot['key_target_history']
    keys = snapshot['exact_finger_key_history']
    if (tips.ndim != 4 or tips.shape[2:] != (10, 3)
            or targets.shape != (*tips.shape[:2], 88, 3)
            or keys.shape != tips.shape[:3] or tips.shape[1] < 1):
        raise ValueError('Fingering reward requires aligned physical-substep geometry and exact keys')
    selected = targets.gather(2, keys.clamp_min(0)[..., None].expand(-1, -1, -1, 3))
    active = keys >= 0
    distances = torch.linalg.vector_norm(tips-selected, dim=-1)
    per_sample = (gaussian_tolerance(distances, tolerance, tolerance*10)*active).sum(-1)
    return (per_sample/active.sum(-1).clamp_min(1)).mean(-1)


@dataclass
class TaskStep:
    observation: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict

    @property
    def done(self):
        return self.terminated | self.truncated


class MusicTaskLogic:
    def __init__(self, bundle, *, num_envs, device, max_episode_steps=None):
        self.manifest = m = bundle.manifest
        self.config = c = m['contract']['environment']
        if c['wrong_press_termination'] or c['disable_fingering_reward']:
            raise ValueError('This profile requires fingering and never terminates on wrong keys')
        if max_episode_steps is not None and (isinstance(max_episode_steps,bool)
                or not isinstance(max_episode_steps,int) or max_episode_steps<1):
            raise ValueError('max_episode_steps must be a positive integer or None')
        self.num_envs, self.device = num_envs, torch.device(device)
        control_dt = float(m.get('control_dt', 0.))
        physics_dt = float(m.get('physics_dt', 0.))
        if control_dt > 0. and physics_dt > 0.:
            decimation = round(control_dt / physics_dt)
            if decimation < 1 or abs(decimation * physics_dt - control_dt) > 1e-8:
                raise ValueError('Task control/physics timesteps must have an integer decimation')
            self.collision_decimation = decimation
        else:
            # Frozen legacy task bundles did not expose the physical timestep;
            # those snapshots use the endpoint form below.
            self.collision_decimation = None
        self.max_episode_steps = max_episode_steps
        self.goal = torch.as_tensor(bundle.goal,device=device,dtype=torch.float32)
        self.fingers = torch.as_tensor(bundle.fingers,device=device,dtype=torch.float32)
        self.observation_goal, self.observation_fingers = self.goal, self.fingers
        self.length = self.goal.shape[0]
        self.lookahead = torch.arange(c['n_steps_lookahead']+1,device=device)
        self.index = torch.zeros(num_envs,device=device,dtype=torch.int64)
        self.done = torch.zeros(num_envs,device=device,dtype=torch.bool)
        self.last_action = torch.zeros((num_envs,51),device=device)
        self.last_reward = torch.zeros(num_envs,device=device)
        self.reset_action_semantics = m.get(
            'reset_action_semantics', 'legacy_action_lower_bound_v1')
        self.reset_action = torch.tensor(m['reset_action'],device=device,dtype=torch.float32)
        if self.reset_action.shape != (51,) or not torch.isfinite(self.reset_action).all():
            raise ValueError('reset_action must contain 51 finite physical controls')
        self.history = torch.zeros((num_envs,m['observation_base_size'],c['frame_stack']),device=device)
        self.metrics_sum = torch.zeros((num_envs,6),device=device)
        self.key_counts = torch.zeros((num_envs,3),device=device,dtype=torch.int64)
        self.returns = torch.zeros(num_envs,device=device)
        self.observation_size = m['contract']['observation_size']

    def reward_terms(self, snapshot, index):
        target = self.goal[index]
        on = target[:,:88].bool()
        wrong = (snapshot['key_activation'] & ~on).any(-1)
        key = key_press_substep_reward(snapshot, target, self.manifest['key_tolerance'])
        sustain = gaussian_tolerance(target[:,-1]-(snapshot['sustain'][:,0]>=.5).float(),
                                     self.manifest['key_tolerance'],self.manifest['key_tolerance']*10)
        finger = fingering_substep_reward(snapshot, self.manifest['finger_tolerance'])
        terms = {'key_press_reward':key,'sustain_reward':sustain,
                 'energy_reward':-self.manifest['energy_coefficient']*snapshot['power'],
                 'fingering_reward':finger}
        if not self.config['disable_forearm_reward']:
            collision = snapshot.get('palm_collision_substeps')
            if collision is not None and self.collision_decimation is not None:
                terms['forearm_reward'] = forearm_contact_reward(
                    collision, self.manifest.get('task_profile', 'tsensei_v1'),
                    decimation=self.collision_decimation)
            else:
                terms['forearm_reward'] = forearm_contact_reward(snapshot['palm_collision'].float(),
                    self.manifest.get('task_profile', 'tsensei_v1'))
        return terms, wrong

    def frame(self, snapshot):
        # At completion keep the final pre-step score frame/window, as Composer does.
        start = self.index.clamp_max(self.length-1)
        idx = start[:,None]+self.lookahead[None,:]
        goal = self.observation_goal[idx.clamp_max(self.length-1)]*(idx<self.length)[:,:,None]
        fields = {'action':self.last_action, 'reward':self.last_reward[:,None],
                  'goal':goal.flatten(1),
                  'fingering':(self.observation_fingers[start].sum(-1)>0).float(),
                  'left_/joints_pos':snapshot['left_qpos'],
                  'right_/joints_pos':snapshot['right_qpos'],
                  'piano/state':snapshot['key_state'],'piano/sustain_state':snapshot['sustain']}
        return torch.cat([fields[x['path']] for x in self.manifest['contract']['observation_fields_before_stack']],dim=1)

    def observation(self):
        # Own the returned memory: a later reset/step must not overwrite a replay transition.
        return self.history.flatten(1).clone()

    def reset(self, ids, snapshot):
        self.index[ids] = 0
        self.done[ids] = False
        self.last_action[ids] = self.reset_action
        self.last_reward[ids] = self.manifest['reset_reward']
        self.history[ids] = 0
        self.history[ids,:,-1] = self.frame(snapshot)[ids]
        self.metrics_sum[ids] = 0
        self.key_counts[ids] = 0
        self.returns[ids] = 0
        return self.observation()

    def advance(self, physical_action, snapshot, *, extra_terms=None, reward_transform=None):
        # Caller validates readiness before stepping physics.
        # Freeze the goal the policy saw before stepping, as Shadow does.
        applied_index = self.index.clone()
        target = self.goal[applied_index]
        terms, wrong = self.reward_terms(snapshot,applied_index)
        if reward_transform is None:
            reward = torch.stack([terms[k] for k in self.manifest['reward_terms']],dim=1).sum(1)
        else:
            terms = reward_transform(terms,physical_action,snapshot,target,
                self.fingers[applied_index])
            reward = (torch.stack(list(terms.values()),dim=1).sum(1) if terms
                      else torch.zeros(self.num_envs,device=self.device))
        if extra_terms:
            if terms.keys() & extra_terms.keys():
                raise ValueError('Extra rewards cannot replace baseline terms')
            reward = reward + torch.stack(list(extra_terms.values()), dim=1).sum(1)
            terms.update(extra_terms)
        self.last_action[:] = physical_action
        self.last_reward[:] = reward
        self.index += 1
        terminated = self.index >= self.length
        truncated = torch.zeros_like(terminated)
        if self.max_episode_steps is not None:
            truncated = (self.index>=self.max_episode_steps) & ~terminated
        self.done[:] = terminated | truncated
        self.history = torch.cat((self.history[:,:,1:],self.frame(snapshot)[:,:,None]),dim=2)
        key_metrics, counts = binary_metrics(target[:,:88],snapshot['key_activation'])
        pedal_metrics,_ = binary_metrics(target[:,-1:],snapshot['sustain']>=.5)
        self.metrics_sum += torch.cat((key_metrics,pedal_metrics),dim=1)
        self.key_counts += counts
        self.returns += reward
        tp,fp,fn = self.key_counts.unbind(1)
        # Existing evaluator's aggregate micro-F1 uses max(denominator, 1).
        micro_f1 = 2*tp.float()/(2*tp+fp+fn).clamp_min(1)
        return TaskStep(self.observation(),reward,torch.ones_like(reward),terminated,truncated,
            {'reward_terms':terms,'score_index':applied_index,'next_score_index':self.index.clone(),
             'goal_applied':target.clone(),'key_activation':snapshot['key_activation'].clone(),
             'wrong_key':wrong,'episode_return':self.returns.clone(),
             'episode_steps':self.index.clone(),
             'frame_metrics':torch.cat((key_metrics,pedal_metrics),dim=1),
             'episode_metrics':self.metrics_sum/self.index[:,None],
             'key_frame_micro_f1':micro_f1,
             'end_reason':terminated.to(torch.int64)+2*truncated.to(torch.int64)})
