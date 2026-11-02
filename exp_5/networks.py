import torch
from torch import nn
from exp_3.networks import cost
from exp_5.safety import barrier


def actor_update(actor, model, transform, optimizer, batch, config, bounds):
    history, last, _, _, reference = batch
    optimizer.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    try:
        action = actor(history, last, transform)
        cv = transform.cv_levels(last, model(history, last, action, transform))
        economic = cost(cv, transform.frequency).mean()
        window = ((action-reference.detach()) / config['regularization_scale']).square().mean()
        center = action.new_tensor(bounds['center'])
        radius = action.new_tensor(bounds[f"p{config.get('threshold_percentile',90)}_radius"])
        cv_barrier = barrier(cv, center[:41], radius[:41], config['barrier_epsilon'])
        mv_barrier = barrier(action, center[41:], radius[41:], config['barrier_epsilon'])
        weighted_window = config['regularization_weight'] * window
        weighted_barrier = config['barrier_weight'] * (cv_barrier + mv_barrier)
        loss = economic + weighted_window + weighted_barrier
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite actor loss')
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    finally:
        model.requires_grad_(True)
    return {k: float(v.detach()) for k,v in dict(actor_loss=loss, actor_cost=economic,
            actor_penalty=window, actor_regularization=weighted_window,
            cv_barrier=cv_barrier, mv_barrier=mv_barrier, actor_barrier=weighted_barrier).items()}
