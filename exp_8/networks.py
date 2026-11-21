"""A shared frozen 53-output forecaster, deterministic actor and Q critic."""
import torch
from torch import nn
from exp_3.networks import Transform, source_model, cost
from exp_8.safety import barrier_per_sample


FEATURES = 53 + 52 + 2
HORIZONS = (1, 5, 10)


class ForecastFeatures(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        config = checkpoint['config']
        if config.get('horizon') != 1 or config.get('target_transform') != 'first_difference':
            raise ValueError('Expected a one-step predictor of all 53 first differences')
        self.transform = Transform(checkpoint)
        if not torch.isfinite(self.transform.mean).all() or not torch.isfinite(self.transform.scale).all():
            raise ValueError('Predictor normalization must be finite')
        if (self.transform.scale <= 0).any():
            raise ValueError('Predictor scales must be positive')
        self.predictor = source_model(checkpoint)
        self.requires_grad_(False)
        self.eval()

    def forecast(self, levels, action):
        return self.predictor(self.transform.history(levels))[:, None]

    def forward(self, levels, lower, upper, remaining, config, nominal=None):
        delta = self.forecast(levels, nominal)[:, 0]
        last = levels[:, -1]
        forecast = last + delta * self.transform.scale + self.transform.mean
        # Bounds express physical levels on an episode-specific scale.
        width = (upper - lower).clamp_min(1e-3)
        state = (last[:, :52] - (lower + upper) / 2) / width
        price = (cost(last[:, :41], self.transform.frequency) - config['target_cost']) / config['cost_scale']
        clock = remaining / config['episode_steps']
        features = torch.cat((delta, state, price[:, None], clock[:, None]), dim=-1)
        return features, forecast


class ControlledForecast(ForecastFeatures):
    """Forecast levels after one explicit command followed by internal control."""
    def __init__(self, checkpoint, conditioned):
        super().__init__(checkpoint)
        self.conditioned = conditioned
        hidden = checkpoint['config']['hidden_size']
        self.future_head = nn.Linear(hidden, 53 * (len(HORIZONS) - 1))
        self.command_head = nn.Sequential(nn.Linear(hidden + 11, hidden), nn.Tanh(),
                                          nn.Linear(hidden, 53 * len(HORIZONS))) if conditioned else None
        if conditioned:
            nn.init.zeros_(self.command_head[-1].weight)
            nn.init.zeros_(self.command_head[-1].bias)
        self.train().requires_grad_(True)

    def encode(self, levels):
        _, hidden = self.predictor.gru(self.transform.history(levels))
        return hidden[-1]

    def predict(self, hidden, last, action):
        one_step = self.predictor.head(hidden)
        later = self.future_head(hidden).reshape(-1, len(HORIZONS) - 1, 53)
        forecast = torch.cat((one_step[:, None], later), dim=1)
        if self.conditioned:
            change = (action[:, :11] - last[:, 41:52]) / 2.0
            residual = self.command_head(torch.cat((hidden, change), dim=-1)).reshape(-1, len(HORIZONS), 53)
            forecast = forecast + residual
        return forecast

    def forecast(self, levels, action):
        if action is None:
            action = levels[:, -1, 41:]
        return self.predict(self.encode(levels), levels[:, -1], action)

    def levels(self, last, predicted):
        horizons = last.new_tensor(HORIZONS).view(1, -1, 1)
        return last[:, None] + predicted * self.transform.scale * horizons.sqrt() + self.transform.mean * horizons


class Actor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, 11))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, nominal, config):
        limit = config['residual_limit'] * config.get('correction_gain', 1.0)
        residual = limit * torch.tanh(self.head(features))
        return (nominal[:, :11] + residual).clamp(0, 100)


class Critic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(FEATURES + 11, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, features, nominal, action, config):
        change = (action - nominal[:, :11]) / config['residual_limit']
        return self.head(torch.cat((features, change), dim=-1)).squeeze(-1)


def stage_cost(cv, action, lower, upper, config):
    tracking = ((cost(cv, config['frequency_seconds']) - config['target_cost']) / config['cost_scale']).square()
    cv_barrier = barrier_per_sample(cv, lower[:, :41], upper[:, :41], config['barrier_epsilon'])
    mv_barrier = barrier_per_sample(action, lower[:, 41:], upper[:, 41:], config['barrier_epsilon'])
    scale = config['barrier_log_scale']
    cv_barrier = torch.log1p(cv_barrier / scale)
    mv_barrier = torch.log1p(mv_barrier / scale)
    total = config['tracking_weight'] * tracking + config['barrier_weight'] * (cv_barrier + mv_barrier)
    return total, tracking, cv_barrier, mv_barrier


def actor_update(actor, critic, optimizer, features, last, reference, config, nominal):
    if any(parameter.requires_grad for parameter in critic.parameters()):
        raise ValueError('Freeze the critic before actor optimization')
    optimizer.zero_grad(set_to_none=True)
    action = actor(features, nominal, config)
    q_loss = -critic(features, nominal, action, config).mean()
    delta = ((action - last[:, 41:52]) / config['action_scale']).abs().mean()
    window = ((action - reference) / config['regularization_scale']).square().mean()
    movement = config['action_weight'] * delta
    regularization = (config['regularization_weight'] * window
                      + config['residual_weight'] * (action - nominal[:, :11]).square().mean())
    loss = q_loss + movement + regularization
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite actor objective')
    loss.backward()
    norm = nn.utils.clip_grad_norm_(actor.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    values = dict(actor_loss=loss, actor_q_loss=q_loss, actor_delta=delta,
                  actor_movement=movement, actor_window=regularization, actor_grad_norm=norm)
    return {name: float(value.detach()) for name, value in values.items()}
