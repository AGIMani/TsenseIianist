"""Graph-compatible, per-substep inelastic robot joint stop."""
import numpy as np
import warp as wp


@wp.kernel
def project(q: wp.array2d(dtype=wp.float32), v: wp.array2d(dtype=wp.float32),
            warm: wp.array2d(dtype=wp.float32), qadr: wp.array(dtype=wp.int32),
            dadr: wp.array(dtype=wp.int32), lo: wp.array(dtype=wp.float32),
            hi: wp.array(dtype=wp.float32), counts: wp.array2d(dtype=wp.int32),
            worlds: wp.array(dtype=wp.int32)):
    slot, joint = wp.tid()
    world = worlds[slot]
    qi, di = qadr[joint], dadr[joint]
    old, speed = q[world, qi], v[world, di]
    new = wp.clamp(old, lo[joint], hi[joint])
    outward = (new <= lo[joint] and speed < 0.0) or (new >= hi[joint] and speed > 0.0)
    if new != old or outward:
        warm[world, di] = 0.0
        counts[world, joint] = counts[world, joint] + 1
    if outward:
        v[world, di] = 0.0
    q[world, qi] = new


class HardJointLimits:
    def __init__(self, contract, *, num_envs, device):
        self.contract = contract
        rows = contract['joints']
        self.shape = (num_envs, len(rows))
        self.device = device
        def array(key, dtype):
            return wp.array(np.array([r[key] for r in rows]), dtype=dtype, device=device)
        self.arrays = [array('qpos_address', wp.int32), array('dof_address', wp.int32),
                       array('float32_min', wp.float32), array('float32_max', wp.float32)]
        self.counts = wp.zeros(self.shape, dtype=wp.int32, device=device)
        self.worlds = wp.array(np.arange(num_envs), dtype=wp.int32, device=device)

    def apply(self, data, worlds=None):
        worlds = self.worlds if worlds is None else worlds
        wp.launch(project, dim=(len(worlds), self.shape[1]),
                  inputs=[data.qpos, data.qvel, data.qacc_warmstart, *self.arrays, self.counts, worlds],
                  device=self.device)
