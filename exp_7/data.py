from collections import deque
import random
import numpy as np
import torch
from exp_3.simulator import Environment as BaseEnvironment
from exp_7.safety import fit_bounds, cv_violations


class Environment(BaseEnvironment):
    def reset(self,seed):
        count=self.config['warmup_steps']
        if count<self.window+1:raise ValueError('Warmup must cover model history')
        self.warmup=self.backend.reset(seed=seed,warmup_steps=count,window=count-1,
                                       frequency_seconds=self.config['frequency_seconds'])
        self.bounds=fit_bounds(self.warmup)
        self.levels=self.warmup[-(self.window+1):].copy();self.steps=0;self.frozen=0
        self.cv_violation=cv_violations(self.levels[-1],self.bounds)
        self.termination_reason=self.cv_violation
        self.done=bool(self.cv_violation)
        return self.levels.copy()

    def step(self,action):
        if getattr(self,'done',False):
            raise ValueError('Episode has ended; reset before stepping')
        if action is None:
            raise ValueError('exp_7 requires an actor command on every step')
        following,value,terminated,truncated=super().step(action)
        reasons=[]
        if terminated: reasons.append('simulator_frozen')
        self.cv_violation=cv_violations(following[-1],self.bounds)
        if self.cv_violation: reasons.append(self.cv_violation)
        self.termination_reason=';'.join(reasons)
        terminated=terminated or bool(self.cv_violation)
        self.done=terminated or truncated
        return following,value,terminated,truncated and not terminated


class Replay:
    def __init__(self,capacity,seed,window):
        self.rows=deque(maxlen=capacity);self.rng=random.Random(seed);self.window=window

    def add(self,levels,action,following,bounds,remaining=200):
        self.rows.append(tuple(np.asarray(v,dtype=np.float32).copy() for v in
                         (levels,action,following,bounds['lower'],bounds['upper'],remaining)))

    def sample(self,count,transform,safe_only=False):
        candidates=list(self.rows)
        if safe_only:
            candidates=[row for row in candidates if not cv_violations(
                row[0][-1],dict(lower=row[3],upper=row[4]))]
        if not candidates:
            raise ValueError('Replay has no eligible transitions')
        rows=self.rng.choices(candidates,k=count)
        levels,action,following,lower,upper,remaining=(torch.tensor(np.stack(v)) for v in zip(*rows))
        last=levels[:,-1];reference=levels[:,-self.window:,41:52].mean(1)
        return transform.history(levels),last,action,transform.cv_target(last,following),reference,lower,upper,levels,remaining
