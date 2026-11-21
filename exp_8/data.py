"""Long episodes with soft CV constraints and physical shutdown detection."""
import numpy as np
from exp_3.simulator import Environment as BaseEnvironment
from exp_8.safety import fit_bounds, cv_violations


class Environment(BaseEnvironment):
    def reset(self, seed):
        count = self.config['warmup_steps']
        if count < self.window + 1:
            raise ValueError('Warmup must cover the predictor history')
        self.warmup = self.backend.reset(seed=seed, warmup_steps=count, window=count - 1,
                                         frequency_seconds=self.config['frequency_seconds'])
        self.bounds = fit_bounds(self.warmup)
        self.levels = self.warmup[-(self.window + 1):].copy()
        self.steps = self.frozen = 0
        self.done = False
        self.termination_reason = ''
        self.cv_violation = cv_violations(self.levels[-1], self.bounds)
        return self.levels.copy()

    def step(self, action):
        if self.done:
            raise ValueError('Episode has ended; reset before stepping')
        if action is None:
            return self.consume(*self.backend.step())
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Expected 12 finite MV commands')
        if np.any((action[:11] < 0) | (action[:11] > 100)):
            raise ValueError('Active MV must be in [0, 100]')
        if action[11] != self.bounds['fixed_mv11']:
            raise ValueError('MV[11] must retain its warmup value')
        return self.consume(*self.backend.step(action))

    def consume(self, row, value):
        if self.done:
            raise ValueError('Episode has ended; reset before stepping')
        row = np.asarray(row, dtype=np.float64)
        if not np.isfinite(row).all() or not np.isfinite(value):
            raise FloatingPointError('Non-finite simulator observation/cost')
        unchanged = np.all(np.abs(row[:41] - self.levels[-1, :41]) <= self.config['freeze_atol'])
        self.frozen = self.frozen + 1 if unchanged else 0
        self.levels = np.concatenate((self.levels[1:], row[None, :]))
        self.steps += 1
        terminated = self.frozen >= self.config['freeze_steps']
        truncated = self.steps >= self.config['episode_steps'] and not terminated
        following = self.levels.copy()
        self.cv_violation = cv_violations(following[-1], self.bounds)
        self.termination_reason = 'simulator_frozen' if terminated else ''
        self.done = terminated or truncated
        return following, value, terminated, truncated
