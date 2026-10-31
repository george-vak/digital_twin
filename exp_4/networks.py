import math

import torch
from torch import nn

from exp_3.networks import cost


def validate_regularization(config, available_levels):
    window = config["regularization_window"]
    if type(window) is not int or not 1 <= window <= available_levels:
        raise ValueError(f"regularization_window must be an integer in [1, {available_levels}]")
    for key in ("regularization_weight", "regularization_scale"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if config["regularization_weight"] < 0:
        raise ValueError("regularization_weight must be nonnegative")
    if config["regularization_scale"] <= 0:
        raise ValueError("regularization_scale must be positive")


def actor_update(actor, dynamics, transform, optimizer, history, last, reference, weight, scale):
    """Penalize each MV's deviation from its own past window mean."""
    optimizer.zero_grad(set_to_none=True)
    dynamics.requires_grad_(False)
    try:
        action = actor(history, last, transform)
        prediction = transform.cv_levels(last, dynamics(history, last, action, transform))
        actor_cost = cost(prediction, transform.frequency).mean()
        penalty = ((action - reference.detach()) / scale).square().mean()
        regularization = weight * penalty
        loss = actor_cost + regularization
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite actor loss")
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    finally:
        dynamics.requires_grad_(True)
    return dict(actor_loss=float(loss.detach()), actor_cost=float(actor_cost.detach()),
                actor_penalty=float(penalty.detach()), actor_regularization=float(regularization.detach()))
