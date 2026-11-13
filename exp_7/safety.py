import numpy as np
import torch


def fit_bounds(warmup):
    values=np.asarray(warmup,dtype=float)
    if values.ndim!=2 or values.shape[1]!=53 or not np.isfinite(values).all():
        raise ValueError('Expected finite warmup observations (N,53)')
    if np.ptp(values[:,52])>1e-8:
        raise ValueError('MV[11] is not constant during warmup')
    return dict(lower=np.concatenate((np.quantile(values[:,:41],.01,axis=0),np.zeros(11))).tolist(),
                upper=np.concatenate((np.quantile(values[:,:41],.99,axis=0),np.full(11,100.))).tolist(),
                fixed_mv11=float(values[-1,52]),warmup_samples=len(values),
                method='CV warmup P1/P99; MV[0:11] fixed 0–100; MV[11] excluded')


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


def cv_violations(current,bounds):
    values=np.asarray(current)[:41]
    lower=np.asarray(bounds['lower'])[:41];upper=np.asarray(bounds['upper'])[:41]
    tolerance=2*np.finfo(np.float32).eps*np.maximum(1,np.maximum(abs(lower),abs(upper)))
    bad=(values<lower-tolerance)|(values>upper+tolerance)|~np.isfinite(values)
    return ';'.join(f'CV[{j}]' for j in np.flatnonzero(bad))
