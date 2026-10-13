# Torch adaptation of the local JAX SAC (original robopianist-rl, Apache-2.0).
# Original attribution/license: LICENSE.robopianist-rl, docs/sac_source_manifest.json.
"""Torch SAC: paper optimization presets, existing JAX loss/update semantics.

Paper does not fully specify critic widths or actor double-Q reduction; retain
three critic layers and mean actor Q from the local JAX baseline (documented).
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SACConfig:
    actor_hidden: tuple = (256, 256)
    critic_hidden: tuple = (256, 256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    temp_lr: float = 1e-3
    gamma: float = .85
    tau: float = .005
    init_temperature: float = .2
    target_entropy: float | None = None
    critic_dropout: float = .01
    critic_layer_norm: bool = True

    def __post_init__(self):
        for dims in (self.actor_hidden, self.critic_hidden):
            if not dims or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in dims):
                raise ValueError('Hidden dimensions must be positive integers')
        for x in (self.actor_lr, self.critic_lr, self.temp_lr, self.init_temperature):
            if not math.isfinite(x) or x <= 0:
                raise ValueError('Learning rates and temperature must be positive and finite')
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1 or not 0 <= self.critic_dropout < 1:
            raise ValueError('Invalid gamma, tau or dropout')
        if self.target_entropy is not None and not math.isfinite(self.target_entropy):
            raise ValueError('Target entropy must be finite')

    @classmethod
    def paper(cls, stage=1):
        if stage not in (1, 2):
            raise ValueError('Stage must be 1 or 2')
        return cls(actor_lr=3e-4 if stage == 1 else 1e-4, gamma=.85 if stage == 1 else .90)


class LayerNorm(nn.Module):
    """Flax-compatible epsilon and fast variance for migration comparisons."""
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x.square().mean(-1, keepdim=True) - mean.square()).clamp_min(0)
        return (x - mean) * torch.rsqrt(var + 1e-6) * self.weight + self.bias


class MLP(nn.Module):
    def __init__(self, input_size, hidden, *, layer_norm=False, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        for size in hidden:
            self.layers.append(nn.Linear(input_size, size))
            self.norms.append(LayerNorm(size) if layer_norm else nn.Identity())
            input_size = size

    def forward(self, x, *, training=False, generator=None, masks=None):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if training and self.dropout:
                mask = (torch.rand(x.shape, device=x.device, generator=generator) >= self.dropout
                        if masks is None else masks[i])
                x = x * mask / (1 - self.dropout)
            x = F.gelu(norm(x), approximate='tanh')
        return x


class Actor(nn.Module):
    def __init__(self, obs_size, action_size, hidden, action_profile="legacy"):
        super().__init__()
        from ..mjlab_env.action_mapping import policy_bounds
        low, high = (torch.tensor(value) for value in policy_bounds(action_profile, action_size))
        self.register_buffer("action_shift", (low + high) * .5, persistent=False)
        self.register_buffer("action_scale", (high - low) * .5, persistent=False)
        self.base = MLP(obs_size, hidden)
        self.mean = nn.Linear(hidden[-1], action_size)
        self.log_std = nn.Linear(hidden[-1], action_size)

    def forward(self, obs):
        x = self.base(obs)
        return self.mean(x), self.log_std(x).clamp(-20, 2)

    def sample(self, obs, *, generator=None, noise=None):
        mean, log_std = self(obs)
        if noise is None:
            noise = torch.randn(mean.shape, device=mean.device, generator=generator)
        z = mean + log_std.exp() * noise
        # Stable Tanh log-Jacobian, including near-saturated actions.
        log_prob = (-.5 * noise.square() - .5 * math.log(2 * math.pi) - log_std
                    - 2 * (math.log(2) - z - F.softplus(-2 * z))).sum(-1)
        log_prob = log_prob - self.action_scale.log().sum()
        return z.tanh() * self.action_scale + self.action_shift, log_prob

    def deterministic(self, obs):
        return self(obs)[0].tanh() * self.action_scale + self.action_shift


class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size, config):
        super().__init__()
        self.base = MLP(obs_size + action_size, config.critic_hidden,
                        layer_norm=config.critic_layer_norm, dropout=config.critic_dropout)
        self.value = nn.Linear(config.critic_hidden[-1], 1)

    def forward(self, obs, action, **kwargs):
        return self.value(self.base(torch.cat((obs, action), -1), **kwargs)).squeeze(-1)


class TwinQ(nn.Module):
    def __init__(self, obs_size, action_size, config):
        super().__init__()
        self.qs = nn.ModuleList([QNetwork(obs_size, action_size, config) for _ in range(2)])

    def forward(self, obs, action, *, training=False, generator=None, masks=None):
        return torch.stack([q(obs, action, training=training, generator=generator,
                              masks=None if masks is None else masks[i]) for i, q in enumerate(self.qs)])


class SAC:
    def __init__(self, obs_size, action_size, config=None, *, device='cuda:0', seed=0,
                 compile_learner=False, fused_optimizer=False, action_profile='legacy'):
        self.action_profile = action_profile
        self.config = c = config or SACConfig.paper()
        self.device = torch.empty(0, device=device).device
        self.obs_size, self.action_size = obs_size, action_size
        self.compile_learner = compile_learner
        self.fused_optimizer = fused_optimizer
        if fused_optimizer and self.device.type != 'cuda':
            raise ValueError('Fused Adam requires CUDA in this runner')
        self.target_entropy = -.5 * action_size if c.target_entropy is None else c.target_entropy
        # Module constructors and Xavier initialization do not consume caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actor = Actor(obs_size, action_size, c.actor_hidden, action_profile)
            self.critic = TwinQ(obs_size, action_size, c)
            for module in (self.actor, self.critic):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(c.init_temperature), device=self.device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.actor_lr, eps=1e-8, foreach=False, fused=fused_optimizer)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c.critic_lr, eps=1e-8, foreach=False, fused=fused_optimizer)
        self.temp_optimizer = torch.optim.Adam([self.log_temperature], lr=c.temp_lr, eps=1e-8, foreach=False, fused=fused_optimizer)
        self.action_rng = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.update_rng = torch.Generator(device=self.device).manual_seed(seed + 2)
        self.updates = 0
        # Compile tensor objectives, including their autograd backwards. Keep
        # modules unwrapped so old checkpoint parameter names remain valid.
        self._critic_loss = self._critic_objective
        self._actor_loss = self._actor_objective
        if compile_learner:
            self._critic_loss = torch.compile(self._critic_objective, fullgraph=True)
            self._actor_loss = torch.compile(self._actor_objective, fullgraph=True)

    def _critic_objective(self, state, action, reward, discount, next_state,
                          next_noise, target_masks, critic_masks):
        with torch.no_grad():
            next_action, next_lp = self.actor.sample(next_state, noise=next_noise)
            next_q = self.target_critic(next_state, next_action, training=True,
                                       masks=target_masks).min(0).values
            target = reward + self.config.gamma * discount * (next_q - self.temperature * next_lp)
        qs = self.critic(state, action, training=True, masks=critic_masks)
        return (qs - target).square().mean(), qs, target, next_action, next_lp

    def _actor_objective(self, state, noise, masks):
        action, lp = self.actor.sample(state, noise=noise)
        actor_q = self.critic(state, action, training=True, masks=masks).mean(0)
        return (self.temperature.detach() * lp - actor_q).mean(), action, lp

    def _randomness(self, batch_size, supplied):
        # Draw in the original eager order, outside compiled functions. Both
        # paths share RNG semantics; explicit fixture masks remain supported.
        result = dict(supplied or {})
        for key in ('next_noise', 'target_masks', 'critic_masks', 'actor_noise', 'actor_masks'):
            if result.get(key) is not None:
                continue
            if key.endswith('noise'):
                result[key] = torch.randn((batch_size, self.action_size), device=self.device,
                                          generator=self.update_rng)
            elif self.config.critic_dropout:
                result[key] = [[torch.rand((batch_size, width), device=self.device,
                                           generator=self.update_rng) >= self.config.critic_dropout
                                for width in self.config.critic_hidden] for _ in range(2)]
            else:
                result[key] = None
        return result

    @property
    def temperature(self):
        return self.log_temperature.exp()

    @torch.no_grad()
    def act(self, obs, *, deterministic=False):
        return self.actor.deterministic(obs) if deterministic else self.actor.sample(obs, generator=self.action_rng)[0]

    def update(self, batch, *, randomness=None, diagnostics=False, _finite_accumulator=None,
               _fused_polyak=False, _optimizer_step=None):
        """One critic -> Polyak -> actor -> alpha update. End flags do not mask TD.

        Explicit noises/masks are only used by cross-framework numerical tests.
        Ordinary training uses independent device RNG draws for all stochastic calls.
        """
        c = self.config
        random = self._randomness(batch.state.shape[0], randomness)
        critic_loss, qs, target, next_action, next_lp = self._critic_loss(
            batch.state, batch.action, batch.reward, batch.discount, batch.next_state,
            random['next_noise'], random['target_masks'], random['critic_masks'])
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grads = {k: p.grad.detach().clone() for k, p in self.critic.named_parameters()} if diagnostics else None
        (self.critic_optimizer.step() if _optimizer_step is None else _optimizer_step(self.critic_optimizer))
        with torch.no_grad():
            if _fused_polyak:
                destinations = list(self.target_critic.parameters())
                sources = list(self.critic.parameters())
                torch._foreach_mul_(destinations, 1 - c.tau)
                torch._foreach_add_(destinations, sources, alpha=c.tau)
            else:
                for dest, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                    dest.mul_(1 - c.tau).add_(source, alpha=c.tau)
        # Retain gradients through sampled actions, not critic parameters.
        self.critic.requires_grad_(False)
        try:
            actor_loss, action, lp = self._actor_loss(
                batch.state, random['actor_noise'], random['actor_masks'])
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grads = {k: p.grad.detach().clone() for k, p in self.actor.named_parameters()} if diagnostics else None
            (self.actor_optimizer.step() if _optimizer_step is None else _optimizer_step(self.actor_optimizer))
        finally:
            self.critic.requires_grad_(True)
        entropy = -lp.detach().mean()
        alpha_before = self.temperature.detach().clone()
        temperature_loss = self.temperature * (entropy - self.target_entropy)
        self.temp_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        temperature_grad = self.log_temperature.grad.detach().clone() if diagnostics else None
        (self.temp_optimizer.step() if _optimizer_step is None else _optimizer_step(self.temp_optimizer))
        self.updates += 1
        metrics = dict(critic_loss=critic_loss.detach(), q=qs.detach().mean(), actor_loss=actor_loss.detach(),
                       entropy=entropy, temperature=alpha_before, temperature_loss=temperature_loss.detach())
        finite = torch.isfinite(torch.stack(list(metrics.values()))).all()
        if _finite_accumulator is not None:
            # CUDA learner graphs retain every update's health result on device;
            # their public update(count) checks it before returning to training.
            _finite_accumulator.logical_and_(finite)
        elif not finite:
            raise FloatingPointError('Nonfinite SAC update; abort training')
        if diagnostics:
            metrics['diagnostics'] = dict(target=target, next_action=next_action, next_log_prob=next_lp,
                                          actor_action=action.detach(), actor_log_prob=lp.detach(),
                                          critic_gradients=critic_grads, actor_gradients=actor_grads,
                                          temperature_gradient=temperature_grad)
        return metrics

    def state_dict(self):
        return dict(config=asdict(self.config), obs_size=self.obs_size, action_size=self.action_size,
                    action_profile=self.action_profile,
                    actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    target_critic=self.target_critic.state_dict(), log_temperature=self.log_temperature.detach(),
                    actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                    temp_optimizer=self.temp_optimizer.state_dict(), action_rng=self.action_rng.get_state(),
                    update_rng=self.update_rng.get_state(), updates=self.updates)

    def load_state_dict(self, state, *, weights_only=False):
        if state.get('action_profile', 'legacy') != self.action_profile:
            raise ValueError('SAC checkpoint action profile mismatch')
        for key in ('obs_size', 'action_size'):
            if state[key] != getattr(self, key):
                raise ValueError(f'SAC checkpoint mismatch: {key}')
        import json
        if not weights_only and json.dumps(state['config'], sort_keys=True) != json.dumps(asdict(self.config), sort_keys=True):
            raise ValueError('SAC checkpoint configuration mismatch')
        for name in ('actor', 'critic', 'target_critic'):
            getattr(self, name).load_state_dict(state[name])
        if weights_only:
            # Paper Stage2: fresh optimizer/replay and alpha initialized to 0.2.
            return
        with torch.no_grad():
            self.log_temperature.copy_(state['log_temperature'])
        for name in ('actor_optimizer', 'critic_optimizer', 'temp_optimizer'):
            optimizer = getattr(self, name)
            optimizer.load_state_dict(state[name])
            # Runtime acceleration is not an algorithm/checkpoint contract.
            for group in optimizer.param_groups:
                group['fused'], group['foreach'] = self.fused_optimizer, False
                # Graph capture is a runtime choice, not part of saved Adam math.
                group['capturable'] = False
            for values in optimizer.state.values():
                if 'step' in values:
                    values['step'] = values['step'].to(self.device if self.fused_optimizer else 'cpu')
        for name in ('action_rng', 'update_rng'):
            getattr(self, name).set_state(state[name].cpu())
        self.updates = state['updates']
