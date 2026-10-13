"""CPU contact-buffer -> tracker -> reward regression, including the final per-step cap."""
from copy import deepcopy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from tests.test_paper_rewards import only, scene_fixture
from tsensei_v3_2_pianist.mjlab_env.self_collision import SelfCollisionTracker
from tsensei_v3_2_pianist.mjlab_env.reward_config import canonical, contact_options, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
from tsensei_v3_2_pianist.rl_torch.checkpoint import reward_contract

TERM = 'self_collision_reward'


def fixture(diagnostics):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="right_/right_hand_index_MCP_link"><joint type="slide"/><geom name="a" size=".02"/></body>
      <body name="right_/right_hand_mid_MCP_link"><joint type="slide"/><geom name="b" size=".02"/>
        <geom name="b2" size=".01"/></body>
      <body name="right_/right_hand_thumb_MCP_link"><joint type="slide"/><geom name="c" size=".02"/></body>
      <body name="left_/left_hand_index_MCP_link"><joint type="slide"/><geom name="left" size=".02"/></body>
      <body name="right_/base_link"><joint type="slide"/><geom name="palm" size=".02"/></body>
      <body name="right_/right_hand_index_PIP_link"><joint type="slide"/><geom name="same" size=".02"/></body>
      <body name="right_/right_hand_thumb_rotaback_link2"><joint type="slide"/><geom name="thumbrot" size=".02"/></body>
      <body name="piano/key"><joint type="slide"/><geom name="key" size=".02"/></body>
    </worldbody></mujoco>''')
    tracker = SelfCollisionTracker(model, 2, 'cpu', categories=('self', 'inter_hand'),
                                   diagnostics=diagnostics, accumulate_pair_substeps=True)

    def sample(rows, *, physical=True):
        geom = np.full((8, 2), -1, np.int32)
        distance = np.ones(8, np.float32)
        world = np.zeros(8, np.int32)
        for i, (a, b, env, dist) in enumerate(rows):
            geom[i] = [model.geom(a).id, model.geom(b).id]
            distance[i], world[i] = dist, env
        data = SimpleNamespace(naconmax=8, nacon=wp.array([len(rows)], dtype=wp.int32, device='cpu'),
            contact=SimpleNamespace(geom=wp.array(geom, dtype=wp.vec2i, device='cpu'),
                dist=wp.array(distance, dtype=wp.float32, device='cpu'),
                worldid=wp.array(world, dtype=wp.int32, device='cpu')))
        tracker.sample(SimpleNamespace(wp_data=data, wp_device='cpu'), count_substep=physical)
        wp.synchronize_device('cpu')
    return tracker, sample


def reward(tracker, *, weight=.2, dt=.05, maximum=5.):
    cfg = only(TERM)
    cfg.update(scale_by_dt=True, reference_dt=.05)
    cfg['terms'][TERM].update(weight=weight, params=dict(aggregation='substep_sum', maximum_penalty_per_step=maximum))
    scene = scene_fixture()
    scene['control_dt'] = dt
    manager = RewardManager(cfg, scene, {}, num_envs=2, device='cpu')
    snapshot = dict(self_collision_pairs=tracker.count(('self',)),
                    self_collision_pair_substeps=tracker.count_finger_pair_substeps())
    return manager({}, torch.zeros(2, 51), snapshot, torch.zeros(2, 89), torch.zeros(2, 10, 88))[TERM]


@pytest.mark.parametrize('diagnostics', [False, True])
def test_ten_substeps_pair_dedup_multiple_pairs_and_category_filter(diagnostics):
    tracker, sample = fixture(diagnostics)
    for i in range(10):
        contacts = [('a', 'b', 0, -.001), ('a', 'b2', 0, -.002),
                    ('a', 'left', 0, -.001), ('b', 'key', 0, -.001), ('a', 'c', 0, .001)]
        if i == 0:
            contacts += [('a', 'b', 1, -.001), ('a', 'c', 1, -.001)]
        sample(contacts)
    assert tracker.count_finger_pair_substeps().tolist() == [10, 2]
    assert tracker.count(('self',)).tolist() == [1, 2]
    torch.testing.assert_close(reward(tracker), torch.tensor([-2., -.4]))
    torch.testing.assert_close(reward(tracker, weight=.75), torch.tensor([-5., -1.5]))
    # A new endpoint-only pair remains a legacy diagnostic, not an 11th substep.
    sample([('a', 'c', 0, -.001)], physical=False)
    assert tracker.count(('self',)).tolist() == [2, 2]
    assert tracker.count_finger_pair_substeps().tolist() == [10, 2]
    if diagnostics:
        assert tracker.substep_counts[:, 3].tolist() == [10, 10]
        assert tracker.substep_counts[:, 0].tolist() == [10, 1]


def test_released_contact_no_repeat_and_reset_isolation():
    tracker, sample = fixture(False)
    sample([('a', 'b', 0, -.001), ('a', 'b', 1, -.001)])
    for _ in range(9):
        sample([])
    assert tracker.count_finger_pair_substeps().tolist() == [1, 1]
    torch.testing.assert_close(reward(tracker), torch.tensor([-.2, -.2]))
    tracker.clear(torch.tensor([0]))
    assert tracker.count_finger_pair_substeps().tolist() == [0, 1]
    sample([('a', 'b', 0, -.001), ('a', 'b', 1, -.001)])
    assert tracker.count_finger_pair_substeps().tolist() == [1, 2]
    tracker.clear()
    assert tracker.count_finger_pair_substeps().tolist() == [0, 0]


@pytest.mark.parametrize('dt', [.025, .05])
def test_capped_substep_units_are_not_dt_scaled(dt):
    tracker, sample = fixture(False)
    for _ in range(10):
        sample([('a', 'b', 0, -.001)])
    torch.testing.assert_close(reward(tracker, dt=dt), torch.tensor([-2., 0.]))


def test_current_contract_rejects_uncapped_or_union_modes():
    new = canonical(self_collision_weight=.2)
    assert contact_options(new)['self_collision_substep_sum'] is True
    changed = deepcopy(new)
    changed['terms'][TERM]['params']['maximum_penalty_per_step'] = 4.
    assert reward_contract(dict(reward_config=new)) != reward_contract(dict(reward_config=changed))
    disabled = deepcopy(new)
    disabled['terms'][TERM]['enabled'] = False
    assert 'self_collision_substep_sum' not in contact_options(disabled)
    for params in [{}, dict(aggregation='substep_sum'),
                   dict(aggregation='unknown', maximum_penalty_per_step=5.),
                   dict(aggregation='substep_sum', maximum_penalty_per_step=5., extra=1)]:
        bad = deepcopy(new)
        bad['terms'][TERM]['params'] = params
        with pytest.raises(ValueError):
            validate(bad)
    for cap in [-1, True, float('nan'), float('inf')]:
        bad = deepcopy(new)
        bad['terms'][TERM]['params']['maximum_penalty_per_step'] = cap
        with pytest.raises(ValueError):
            validate(bad)


@pytest.mark.parametrize('weight,cap', [(.2, 5.), (.75, 5.), (0., 5.), (.2, 0.), (.2, 3.)])
@pytest.mark.parametrize('dt', [.025, .05, .1])
def test_cap_is_after_weight_once_per_control_step_and_logs_match(weight, cap, dt):
    counts = torch.tensor([0, 1, 10, 20, 24, 25, 26, 50, 1000])
    cfg = only(TERM)
    cfg.update(scale_by_dt=True, reference_dt=.05)
    cfg['terms'][TERM].update(weight=weight)
    cfg['terms'][TERM]['params']['maximum_penalty_per_step'] = cap
    scene = scene_fixture(); scene['control_dt'] = dt
    manager = RewardManager(cfg, scene, {}, num_envs=len(counts), device='cpu')
    # Deliberately different union counts detect any accidental legacy fallback.
    sample = dict(self_collision_pair_substeps=counts, self_collision_pairs=torch.ones_like(counts))
    result = manager({}, torch.zeros(len(counts), 51), sample,
                     torch.zeros(len(counts), 89), torch.zeros(len(counts), 10, 88))[TERM]
    expected = torch.tensor([-min(weight*int(n), cap) for n in counts])
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(manager.raw_terms[TERM], -counts.float())
    torch.testing.assert_close(manager.weighted_terms[TERM], expected)


def test_many_physical_pair_substeps_cap_then_clear_returns_to_zero():
    tracker, sample = fixture(False)
    for _ in range(10):
        sample([('a', 'b', 0, -.001), ('a', 'c', 0, -.001), ('b', 'c', 0, -.001),
                ('a', 'b', 1, -.001)])
    assert tracker.count_finger_pair_substeps().tolist() == [30, 10]
    torch.testing.assert_close(reward(tracker), torch.tensor([-5., -2.]))
    tracker.clear(torch.tensor([0]))
    torch.testing.assert_close(reward(tracker), torch.tensor([0., -2.]))
    tracker.clear()
    torch.testing.assert_close(reward(tracker), torch.zeros(2))


@pytest.mark.parametrize('diagnostics', [False, True])
def test_shell_and_same_digit_contacts_are_diagnostic_only(diagnostics):
    tracker, sample = fixture(diagnostics)
    for _ in range(10):
        sample([('a', 'palm', 0, -.01), ('a', 'same', 0, -.01),
                ('c', 'thumbrot', 0, -.01), ('c', 'palm', 0, -.01)])
    assert tracker.count_pair_substeps().tolist() == [40, 0]
    assert tracker.count(('self',)).tolist() == [4, 0]
    assert tracker.count_finger_pair_substeps().tolist() == [0, 0]
    torch.testing.assert_close(reward(tracker), torch.zeros(2))
    sample([('same', 'thumbrot', 0, -.01), ('same', 'left', 0, -.01)])
    assert tracker.count_finger_pair_substeps().tolist() == [1, 0]
    assert tracker.count(('inter_hand',)).tolist() == [1, 0]
    torch.testing.assert_close(reward(tracker), torch.tensor([-.2, 0.]))


def test_all_actual_urdf_links_have_explicit_digit_mapping():
    from itertools import combinations
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from tsensei_v3_2_pianist.mjlab_env.self_collision import finger_identity, is_distinct_finger_pair
    for side in ('left', 'right'):
        root = ET.parse(Path(__file__).resolve().parents[1] / 'tsensei_v3_2_pianist/assets/urdf' / f'TsenseI_{side}.urdf')
        names = [f"{side}_/{link.get('name')}" for link in root.findall('link')]
        ids = [finger_identity(n) for n in names]
        assert ids.count(None) == 2  # base and palm shell
        for digit in ('thumb', 'index', 'mid', 'ring', 'pinky'):
            assert ids.count((side, digit)) == 4
        for a, b in combinations(names, 2):
            ia, ib = finger_identity(a), finger_identity(b)
            assert is_distinct_finger_pair(a, b) == bool(ia and ib and ia != ib)
    old = canonical()
    old['contract'] = 'shadow_control_frame_current'
    with pytest.raises(ValueError, match='Reward config'):
        validate(old)
