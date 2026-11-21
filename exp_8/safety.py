"""Finite C2 extensions of two-sided logarithmic barriers."""
import torch
from exp_7.safety import fit_bounds, cv_violations


def relaxed_log(slack, epsilon):
    overflow = (epsilon - slack) / epsilon
    return torch.where(slack >= epsilon, -torch.log(slack.clamp_min(epsilon)),
                       -slack.new_tensor(epsilon).log() + overflow + 0.5 * overflow.square())


def barrier_per_sample(values, lower, upper, epsilon=0.1):
    width = upper - lower
    scale = width.clamp_min(1e-3)
    left = 2 * (values - lower) / scale
    right = 2 * (upper - values) / scale
    interval = (relaxed_log(left, epsilon) + relaxed_log(right, epsilon)) / 2
    point = ((values - lower) / 1e-3).square()
    return torch.where(width > 1e-10, interval, point).mean(dim=-1)
