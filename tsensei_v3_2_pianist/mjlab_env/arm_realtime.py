"""Opt-in single-scene CPU implementation of the current ArmFollower algorithm.

Numba is optional (requirements-tested.txt). This is a deployment component,
also reused by the opt-in native scene training backend. Compile/warm up before starting a
control clock; rendering, file I/O and hardware transport belong outside it.
Float64, up to 12 iterations, five candidates, current-step hard checks and
the same endpoint-feasibility bounds as the Torch implementation are kept.
"""
import numpy as np
from numba import njit
from .arm_follow import ArmFollower, SIDES


@njit(cache=True, inline='always')
def _mm3(a, b):
    # Fixed-size products avoid thousands of tiny BLAS calls per IK tick.
    # No fastmath: preserve float64 operation order and all safety checks.
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            out[i,j] = a[i,0]*b[0,j] + a[i,1]*b[1,j] + a[i,2]*b[2,j]
    return out


@njit(cache=True, inline='always')
def _mv3(a, b):
    out = np.empty(3)
    for i in range(3):
        out[i] = a[i,0]*b[0] + a[i,1]*b[1] + a[i,2]*b[2]
    return out


@njit(cache=True)
def _rot(axis, angle):
    x, y, z = axis
    c, s = 1-np.cos(angle), np.sin(angle)
    # Expanded I + sin(q)K + (1-cos(q))K^2, including non-unit axes.
    return np.array([[1-c*(y*y+z*z), c*x*y-s*z, c*x*z+s*y],
                     [c*y*x+s*z, 1-c*(x*x+z*z), c*y*z-s*x],
                     [c*z*x-s*y, c*z*y+s*x, 1-c*(x*x+y*y)]])


@njit(cache=True)
def _fk(q, jp, jr, axes, bp, br, tp, tr, jacobian=True):
    p, r = bp.copy(), br.copy()
    anchor = np.empty((7, 3)); axis = np.empty((7, 3))
    for i in range(7):
        p = p + _mv3(r, jp[i])
        r = _mm3(r, jr[i])
        if jacobian:
            anchor[i] = p; axis[i] = _mv3(r, axes[i])
        r = _mm3(r, _rot(axes[i], q[i]))
    tip = p + _mv3(r, tp)
    jac = np.zeros((6, 7))
    if jacobian:
        for i in range(7):
            d = tip-anchor[i]; a = axis[i]
            jac[0,i] = a[1]*d[2]-a[2]*d[1]
            jac[1,i] = a[2]*d[0]-a[0]*d[2]
            jac[2,i] = a[0]*d[1]-a[1]*d[0]
            jac[3:,i] = a
    return tip, _mm3(r, tr), jac


@njit(cache=True)
def _rotation_error(actual, target):
    r = _mm3(target, actual.T)
    values = np.sqrt(np.maximum(np.array([1+r[0,0]+r[1,1]+r[2,2],
        1+r[0,0]-r[1,1]-r[2,2], 1-r[0,0]+r[1,1]-r[2,2],
        1-r[0,0]-r[1,1]+r[2,2]]), 0.))
    i = np.argmax(values)
    if i == 0:
        q = np.array([values[0]**2, r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])
    elif i == 1:
        q = np.array([r[2,1]-r[1,2], values[1]**2, r[1,0]+r[0,1], r[0,2]+r[2,0]])
    elif i == 2:
        q = np.array([r[0,2]-r[2,0], r[1,0]+r[0,1], values[2]**2, r[2,1]+r[1,2]])
    else:
        q = np.array([r[1,0]-r[0,1], r[2,0]+r[0,2], r[2,1]+r[1,2], values[3]**2])
    q /= 2*max(values[i], 1e-8)
    q /= np.sqrt(np.sum(q*q))
    if q[0] < 0: q = -q
    length = np.sqrt(np.sum(q[1:]*q[1:]))
    return q[1:]*(2*np.arctan2(length, max(q[0], 0.))/max(length, 1e-12))


@njit(cache=True)
def _error(p, r, tp, tr, weight):
    e = np.empty(6)
    e[:3] = tp-p; e[3:] = weight*_rotation_error(r, tr)
    return e


@njit(cache=True)
def _coupling(q, j6, j7, coefficient):
    return coefficient*abs(q[j6])+abs(q[j7])


@njit(cache=True)
def _segment(start, candidate, j6, j7, coefficient, limit):
    # Same 48-step feasible-prefix bisection as the reference; fast-pathing
    # already-feasible endpoints changes no result and removes idle work.
    if _coupling(candidate, j6, j7, coefficient) <= limit+1e-12:
        return candidate
    lo, hi = 0., 1.
    delta = candidate-start
    for _ in range(48):
        mid = (lo+hi)*.5
        if _coupling(start+mid*delta, j6, j7, coefficient) <= limit+1e-12:
            lo = mid
        else:
            hi = mid
    return start+lo*delta


@njit(cache=True)
def _stop(gap, amax, dt):
    delta = amax*dt*(1-1e-7)
    gap = np.maximum(gap, 0.)
    n = np.floor((np.sqrt(1+8*gap/(delta*dt))-1)/2)
    return (gap/dt+delta*n*(n+1)/2)/(n+1)


@njit(cache=True)
def _advance(q0, v0, reference, tp, tr, jp, jr, axes, bp, br, toolp, toolr,
             low, high, vmax, amax, dt, iterations, weight, damping,
             j6, j7, coefficient, limit, soft):
    dv = amax*dt*(1-1e-10)
    vlo = np.maximum(-vmax, v0-dv); vhi = np.minimum(vmax, v0+dv)
    vlo = np.maximum(vlo, -_stop(q0-low, amax, dt))
    vhi = np.minimum(vhi, _stop(high-q0, amax, dt))
    lo = np.maximum(low, q0+vlo*dt); hi = np.minimum(high, q0+vhi*dt)
    if np.any(lo > hi): raise RuntimeError('Arm has no feasible hard-limit interval; state not advanced')
    q = np.minimum(np.maximum(reference, low), high)
    fractions = (1., .5, .25, .125, 0.)
    for _ in range(iterations):
        p, r, jac = _fk(q, jp, jr, axes, bp, br, toolp, toolr)
        e = _error(p, r, tp, tr, weight)
        # Numerical convergence of the IK reference only. Actual execution
        # still goes through every braking and hard-limit check below.
        # This weighted pose residual is not the requested 1 mm tracking bound.
        if np.sum(e*e) <= 1e-30:
            break
        jac[3:, :] *= weight
        delta = jac.T@np.linalg.solve(jac@jac.T+damping*damping*np.eye(6), e)
        best = np.inf
        chosen = q.copy()
        for fraction in fractions:
            candidate = np.minimum(np.maximum(q+fraction*delta, low), high)
            if _coupling(candidate, j6, j7, coefficient) > limit+1e-12: continue
            cp, cr, _ = _fk(candidate, jp, jr, axes, bp, br, toolp, toolr, False)
            ce = _error(cp, cr, tp, tr, weight)
            cost = np.sum(ce*ce)
            if cost < best:
                best, chosen = cost, candidate
        q = chosen
    requested_v = (q-q0)/dt; requested_a = (requested_v-v0)/dt
    excess = max(0., np.max(np.abs(requested_v)/(vmax*soft)-1),
                 np.max(np.abs(requested_a)/(amax*soft)-1))
    # Follow the current IK reference as far as the current-step feasible box
    # permits. There is no remaining-distance target-braking rule.
    issued = np.minimum(np.maximum(q, lo), hi)
    issued = _segment(q0, issued, j6, j7, coefficient, limit)
    v = (issued-q0)/dt; acc = (v-v0)/dt
    if not np.all(np.isfinite(issued)) or not np.all(np.isfinite(q)):
        raise FloatingPointError('Arm IK numerical failure; state not advanced')
    if np.any(np.abs(v)>vmax+1e-8) or np.any(np.abs(acc)>amax+1e-7):
        raise RuntimeError('Arm hard-limit verification failed; state not advanced')
    if np.any(issued<low-1e-12) or np.any(issued>high+1e-12):
        raise RuntimeError('Arm position verification failed; state not advanced')
    if _coupling(issued, j6, j7, coefficient)>limit+1e-12:
        raise RuntimeError('Arm CCS-696 coupling verification failed; state not advanced')
    p, r, _ = _fk(issued, jp, jr, axes, bp, br, toolp, toolr, False)
    return issued, v, acc, q, p, r, requested_v, requested_a, excess


class RealtimeArmFollower:
    """Single dual-arm CPU loop; explicit opt-in, no hardware commands.

    targets maps side to (world position [3], world rotation [3,3]). A failed
    check on either arm leaves BOTH arms' q/v/reference unchanged. Construct
    from approved current options and installed poses, then call warmup().
    The caller owns its 5 ms deadline and should separately monitor overruns.
    """
    def __init__(self, options, initial_poses, *, dt=.005):
        # Reuse canonical source/limit validation and deterministic calibration.
        reference = ArmFollower(options, initial_poses, num_envs=1, dt=dt, device='cpu')
        self.options, self.dt = reference.options, reference.dt
        self.contract = dict(reference.contract, execution_backend='numba_float64_single_scene')
        self.sides = {s:{k:np.ascontiguousarray(v.numpy()).copy() for k,v in a.items()}
                      for s,a in reference.sides.items()}
        for a in self.sides.values():
            for k in ('q','v','acceleration','reference'): a[k] = a[k][0].copy()
        self.initial_poses = {s:(np.array(p, dtype=np.float64),np.array(r, dtype=np.float64))
                              for s,(p,r) in initial_poses.items()}
        self.last = {}; self.excess = 0.

    def reset(self):
        for a in self.sides.values():
            a['q'][:] = a['q0']; a['reference'][:] = a['q0']
            a['v'][:] = 0; a['acceleration'][:] = 0
        self.last = {}; self.excess = 0.

    def warmup(self):
        """Compile and run once before enabling the control clock, then reset."""
        self.step(self.initial_poses)
        self.reset()

    def begin_control(self):
        self.excess = 0.

    def step(self, targets):
        pending = {}; diagnostics = {}
        o = self.options; c = o['coupling']
        for side, a in self.sides.items():
            tp, tr = [np.ascontiguousarray(x, dtype=np.float64) for x in targets[side]]
            if tp.shape != (3,) or tr.shape != (3,3) or not (np.isfinite(tp).all() and np.isfinite(tr).all()):
                raise ValueError('Invalid palm pose request: '+side)
            result = _advance(a['q'],a['v'],a['reference'],tp,tr,
                *[a[k] for k in ('joint_pos','joint_rot','axis','base_pos','base_rot','tool_pos','tool_rot',
                                 'low','high','vmax','amax')],
                self.dt,o['iterations'],o['orientation_weight_m'],o['damping'],
                c['j6_index'],c['j7_index'],c['coefficient'],c['sum_limit_rad'],o['soft_limit_ratio'])
            q,v,acc,raw,p,r,rv,ra,excess = result
            rotation = _rotation_error(r,tr)
            diagnostics[side] = dict(target_position=tp.copy(),target_rotation=tr.copy(),
                requested_q=raw,requested_velocity=rv,requested_acceleration=ra,
                position_error=np.linalg.norm(tp-p),orientation_error=np.linalg.norm(rotation),
                position_error_vector=tp-p,rotation_error_vector=rotation,excess=excess)
            pending[side] = result
        for side, result in pending.items():
            a = self.sides[side]
            for key,value in zip(('q','v','acceleration','reference'),result[:4]): a[key][:] = value
            self.excess = max(self.excess,result[-1])
        self.last = diagnostics
        return {s:(r[4],r[5]) for s,r in pending.items()}

    def observation(self):
        values=[]
        for s,a in self.sides.items():
            values.extend(a[k] for k in ('q','v','reference'))
            values.extend(self.last[s][k] if s in self.last else np.zeros(3)
                          for k in ('position_error_vector','rotation_error_vector'))
        return np.concatenate(values).astype(np.float32)

    def penalty(self):
        return -self.options['maximum_penalty_per_step']*min(1.,self.excess/self.options['full_penalty_excess'])
