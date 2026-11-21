"""Fit and validate Q using matched simulator branches."""
import copy

import torch
from torch import nn

from exp_8.networks import Critic, ControlledForecast, stage_cost


@torch.no_grad()
def inputs(encoder, data, config):
    features = []
    remaining = data['remaining'].clamp_min(1)
    for start in range(0, len(remaining), 64):
        batch = slice(start, start + 64)
        nominal = data['nominal'][batch] if isinstance(encoder, ControlledForecast) else None
        vector, _ = encoder(data['levels'][batch], data['lower'][batch], data['upper'][batch],
                            remaining[batch], config, nominal)
        features.append(vector)
    levels = data['levels']
    return dict(features=torch.cat(features), last=levels[:, -1],
                reference=levels[:, -config['regularization_window']:, 41:52].mean(dim=1),
                nominal=data['nominal'], actions=data['actions'][..., :11])


def returns(data, config):
    future = data['future']
    size, branches, horizon, _ = future.shape
    lower = data['lower'][:, None, None].expand(size, branches, horizon, 52).reshape(-1, 52)
    upper = data['upper'][:, None, None].expand(size, branches, horizon, 52).reshape(-1, 52)
    total, _, _, _ = stage_cost(future[..., :41].reshape(-1, 41), future[..., 41:52].reshape(-1, 11),
                               lower, upper, config)
    total = total.reshape(size, branches, horizon)
    # On branch trajectories, treat three identical CV transitions as absorbing.
    previous = torch.cat((data['levels'][:, None, -1:, :41].expand(-1, branches, -1, -1),
                          future[..., :-1, :41]), dim=-2)
    unchanged = (future[..., :41] == previous).all(dim=-1)
    frozen = unchanged[..., 2:] & unchanged[..., 1:-1] & unchanged[..., :-2]
    total[..., 2:] = torch.where(frozen.cumsum(dim=-1) > 0, config['failure_penalty'], total[..., 2:])
    weights = config['gamma'] ** torch.arange(horizon)
    weights = weights / weights.sum()
    return -(total * weights).sum(dim=-1)


def predict(critic, data, indices, config):
    count = data['actions'].shape[1]
    features = data['features'][indices].repeat_interleave(count, dim=0)
    nominal = data['nominal'][indices].repeat_interleave(count, dim=0)
    actions = data['actions'][indices].flatten(0, 1)
    return critic(features, nominal, actions, config).reshape(-1, count)


def metrics(prediction, target):
    advantage = prediction - prediction[:, :1]
    truth = target - target[:, :1]
    pairs = torch.triu_indices(target.shape[1], target.shape[1], offset=1)
    gap = target[:, pairs[0]] - target[:, pairs[1]]
    difference = prediction[:, pairs[0]] - prediction[:, pairs[1]]
    valid = gap.abs() > 1e-5
    chosen = target.gather(1, prediction.argmax(dim=1, keepdim=True)).squeeze(1)
    regret = target.max(dim=1).values - chosen
    return dict(mse=float((prediction - target).square().mean()),
                advantage_rmse=float((advantage - truth).square().mean().sqrt()),
                pair_accuracy=float((gap[valid].sign() == difference[valid].sign()).float().mean()),
                pairs=int(valid.sum()), regret=float(regret.mean()))


def fit_critic(encoder, train, validation, config):
    torch.manual_seed(config['seed'])
    critic = Critic(config['hidden_size'])
    replay = inputs(encoder, train, config)
    heldout = inputs(encoder, validation, config)
    target = returns(train, config)
    heldout_target = returns(validation, config)
    scale = (target - target[:, :1]).square().mean().sqrt().clamp_min(0.1)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=config['critic_lr'])
    best, best_state = float('inf'), None
    history = []
    for update in range(1, config['critic_updates'] + 1):
        indices = torch.randint(len(target), (config['batch_size'],))
        prediction = predict(critic, replay, indices, config)
        truth = target[indices] / scale
        value_loss = nn.functional.smooth_l1_loss(prediction, truth)
        advantage_loss = nn.functional.smooth_l1_loss(prediction - prediction[:, :1], truth - truth[:, :1])
        optimizer.zero_grad(set_to_none=True)
        (value_loss + 3 * advantage_loss).backward()
        nn.utils.clip_grad_norm_(critic.parameters(), config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        if update % 100 == 0 or update == config['critic_updates']:
            with torch.no_grad():
                fitted = predict(critic, heldout, torch.arange(len(heldout_target)), config) * scale
                diagnostic = metrics(fitted, heldout_target)
            history.append(dict(update=update, **diagnostic))
            if diagnostic['regret'] < best:
                best, best_state = diagnostic['regret'], copy.deepcopy(critic.state_dict())
    critic.load_state_dict(best_state)
    critic.zero_grad(set_to_none=True)
    critic.eval().requires_grad_(False)
    with torch.no_grad():
        fitted = predict(critic, heldout, torch.arange(len(heldout_target)), config) * scale
    diagnostic = metrics(fitted, heldout_target)
    diagnostic.update(return_scale=float(scale), history=history,
                      nominal_regret=float((heldout_target.max(dim=1).values - heldout_target[:, 0]).mean()),
                      train_anchors=len(target), validation_anchors=len(heldout_target))
    return critic, replay, diagnostic
