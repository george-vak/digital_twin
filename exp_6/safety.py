import numpy as np
import torch


def fit_bounds(warmup):
    values=np.asarray(warmup,dtype=float)
    if values.ndim!=2 or values.shape[1]!=53 or not np.isfinite(values).all():
        raise ValueError('Expected finite warmup observations (N,53)')
    if np.ptp(values[:,52])>1e-8:
        raise ValueError('MV[11] is not constant during warmup')
    return dict(lower=np.quantile(values[:,:52],.01,axis=0).tolist(),
                upper=np.quantile(values[:,:52],.99,axis=0).tolist(),
                fixed_mv11=float(values[-1,52]),warmup_samples=len(values),
                method='raw per-channel warmup P1/P99; MV[11] excluded')


def barrier(values, lower, upper, epsilon=.1):
    width=upper-lower
    # A zero-width CV interval remains a point constraint. The floor is only
    # a loss scale and does not widen the gate's physical thresholds.
    scale=width.clamp_min(1e-3)
    slack=2*torch.minimum(values-lower,upper-values)/scale
    overflow=(epsilon-slack)/epsilon
    relaxed=torch.where(slack>=epsilon,-torch.log(slack.clamp_min(epsilon)),
                        -values.new_tensor(epsilon).log()+overflow+.5*overflow.square())
    point=((values-lower)/1e-3).square()
    return torch.where(width>1e-10,relaxed,point).mean()


class ReleaseGate:
    def __init__(self,bounds):
        self.lower=np.asarray(bounds['lower']);self.upper=np.asarray(bounds['upper'])
        # Account only for float32 rounding when comparing physical commands.
        self.tolerance=2*np.finfo(np.float32).eps*np.maximum(1,np.maximum(abs(self.lower),abs(self.upper)))

    def decide(self,current,candidate):
        current=np.asarray(current)[:52]; candidate=np.asarray(candidate)
        if candidate.shape!=(11,):raise ValueError('Expected 11 active MV commands')
        bad=(current<self.lower-self.tolerance)|(current>self.upper+self.tolerance)|~np.isfinite(current)
        action_bad=(candidate<self.lower[41:]-self.tolerance[41:])|(candidate>self.upper[41:]+self.tolerance[41:])|~np.isfinite(candidate)
        reasons=[f'{"CV" if j<41 else "MV"}[{j if j<41 else j-41}]' for j in np.flatnonzero(bad)]
        reasons += [f'candidate_MV[{j}]' for j in np.flatnonzero(action_bad)]
        return bool(reasons),';'.join(reasons)
