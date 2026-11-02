"""Fixed marginal deviation quantiles, relaxed log barrier and release gate."""
import numpy as np
import torch


def fit_bounds(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 53 or not np.isfinite(values).all():
        raise ValueError('Expected finite physical CV/MV observations of shape (N, 53)')
    center = np.median(values, axis=0)
    deviations = np.abs(values - center)
    floor = np.maximum(np.abs(center) * 1e-4, 1e-3)
    inner = np.maximum(np.quantile(deviations, .8, axis=0), floor)
    outer = np.maximum(np.quantile(deviations, .9, axis=0), inner + floor)
    return dict(center=center.tolist(), p80_radius=inner.tolist(), p90_radius=outer.tolist(),
                method='per-channel absolute deviation from median; fixed calibration set',
                radius_floor=floor.tolist())


def barrier(values, center, radius, epsilon=.1):
    """-log(distance to boundary), with a finite C1 quadratic extension outside."""
    slack = 1 - (values - center).abs() / radius
    log_part = -torch.log(slack.clamp_min(epsilon))
    overflow = (epsilon - slack).clamp_min(0) / epsilon
    return (log_part + overflow + .5 * overflow.square()).mean()


class ReleaseGate:
    def __init__(self, bounds, min_release_steps=5, recovery_steps=3, percentile=None):
        self.center = np.asarray(bounds['center'])
        if percentile not in (None, 80, 90):
            raise ValueError('percentile must be 80 or 90')
        # None preserves the initial diagnostic P90/P80 hysteresis run.
        self.inner = np.asarray(bounds[f'p{80 if percentile is None else percentile}_radius'])
        self.outer = np.asarray(bounds[f'p{90 if percentile is None else percentile}_radius'])
        self.min_release_steps = min_release_steps
        self.recovery_steps = recovery_steps
        self.released = False
        self.elapsed = self.stable = 0

    def decide(self, current, candidate):
        # Check actual CV/MV and the candidate MV including exploration noise.
        current_bad = np.abs(current - self.center) > self.outer
        action_bad = np.abs(candidate - self.center[41:]) > self.outer[41:]
        reasons = [f'{"CV" if j < 41 else "MV"}[{j if j < 41 else j-41}]'
                   for j in np.flatnonzero(current_bad)]
        reasons += [f'candidate_MV[{j}]' for j in np.flatnonzero(action_bad)]
        unsafe = bool(reasons)
        if not self.released and unsafe:
            self.released = True
            self.elapsed = self.stable = 0
        if self.released:
            recovered = bool(np.all(np.abs(current - self.center) <= self.inner)) and not unsafe
            self.stable = self.stable + 1 if recovered else 0
            if self.elapsed >= self.min_release_steps and self.stable >= self.recovery_steps:
                self.released = False
                self.elapsed = self.stable = 0
            else:
                self.elapsed += 1
        return self.released, ';'.join(reasons) if reasons else ('recovery_wait' if self.released else '')
