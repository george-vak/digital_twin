from collections import deque
import random
import numpy as np
import torch
from exp_3.simulator import Environment as BaseEnvironment
from exp_6.safety import fit_bounds


class Environment(BaseEnvironment):
    def reset(self,seed):
        count=self.config['warmup_steps']
        if count<self.window+1:raise ValueError('Warmup must cover model history')
        self.warmup=self.backend.reset(seed=seed,warmup_steps=count,window=count-1,
                                       frequency_seconds=self.config['frequency_seconds'])
        self.bounds=fit_bounds(self.warmup)
        self.levels=self.warmup[-(self.window+1):].copy();self.steps=0;self.frozen=0
        return self.levels.copy()


class Replay:
    def __init__(self,capacity,seed,window):
        self.rows=deque(maxlen=capacity);self.rng=random.Random(seed);self.window=window

    def add(self,levels,action,following,bounds):
        self.rows.append(tuple(np.asarray(v,dtype=np.float32).copy() for v in
                         (levels,action,following,bounds['lower'],bounds['upper'])))

    def sample(self,count,transform):
        rows=self.rng.sample(list(self.rows),count)
        levels,action,following,lower,upper=(torch.tensor(np.stack(v)) for v in zip(*rows))
        last=levels[:,-1];reference=levels[:,-self.window:,41:52].mean(1)
        return transform.history(levels),last,action,transform.cv_target(last,following),reference,lower,upper
