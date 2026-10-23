"""CPU Warp regression: mesh-centroid transforms never enter key targets."""
from types import SimpleNamespace
import numpy as np
import torch
import warp as wp
from tsensei_v3_2_pianist.mjlab_env.substep_rewards import SubstepRewardCollector
from tests.test_substep_rewards import fixture


def test_each_substep_samples_the_reference_sites_not_mesh_centroids():
    _, sim = fixture()
    positions = np.arange(2*98*3, dtype=np.float32).reshape(2,98,3)*.001
    sim.wp_data.site_xpos = wp.array(positions,dtype=wp.vec3,device='cpu')
    # Deliberately unrelated principal-axis mesh centers: a geom read must fail.
    sim.wp_data.geom_xpos = wp.array(positions+100,dtype=wp.vec3,device='cpu')
    offsets = np.tile(np.array([.025,0,.0045],np.float32),(88,1))
    tracker=SubstepRewardCollector(num_envs=2,decimation=10,device='cpu',
        key_qadr=[0,2],key_limits=[[0,.1],[0,.2]],action_ids=[0,2],
        geom_pair_mask=np.zeros((4,4),bool),tip_ids=np.arange(10),
        key_site_ids=np.arange(10,98),key_offsets=offsets)
    for step in range(10):
        wp.to_torch(sim.wp_data.site_xpos)[:] = torch.from_numpy(positions+step*.01)
        tracker.sample(sim,step)
    wp.synchronize_device('cpu')
    for step in range(10):
        np.testing.assert_allclose(tracker.fingertip_history[:,step],positions[:,:10]+step*.01,atol=1e-7)
        np.testing.assert_allclose(tracker.key_target_history[:,step],positions[:,10:]+step*.01+offsets,atol=1e-7)
    other=tracker.key_target_history[1].clone()
    tracker.clear([0])
    assert not tracker.key_target_history[0].any()
    torch.testing.assert_close(tracker.key_target_history[1],other,rtol=0,atol=0)
