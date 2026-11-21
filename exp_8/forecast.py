"""Compare the source forecast with GRUs adapted to controlled trajectories."""
import copy

import torch
from torch import nn

from exp_3.networks import cost
from exp_8.networks import ControlledForecast, ForecastFeatures, HORIZONS


def forecast_metrics(prediction, target, frequency):
    error = prediction - target
    price_error = cost(prediction[..., :41], frequency) - cost(target[..., :41], frequency)
    return {str(horizon): dict(cv_mae=float(error[:, index, :41].abs().mean()),
                               mv_mae=float(error[:, index, 41:].abs().mean()),
                               price_rmse=float(price_error[:, index].square().mean().sqrt()))
            for index, horizon in enumerate(HORIZONS)}


def predict_levels(model, levels, actions):
    if isinstance(model, ControlledForecast):
        return model.levels(levels[:, -1], model.forecast(levels, actions))
    # Recursively apply the original one-step model on its own predictions.
    answers = []
    for step in range(1, max(HORIZONS) + 1):
        scaled = model.predictor(model.transform.history(levels))
        following = levels[:, -1] + scaled * model.transform.scale + model.transform.mean
        levels = torch.cat((levels[:, 1:], following[:, None]), dim=1)
        if step in HORIZONS:
            answers.append(following)
    return torch.stack(answers, dim=1)


@torch.no_grad()
def evaluate_forecast(model, data):
    count, branches = data['actions'].shape[:2]
    chunks = []
    for start in range(0, count, 32):
        levels = data['levels'][start:start + 32].repeat_interleave(branches, dim=0)
        actions = data['actions'][start:start + 32].flatten(0, 1)
        chunks.append(predict_levels(model, levels, actions))
    actual = data['future'][:, :, [h - 1 for h in HORIZONS]].flatten(0, 1)
    return forecast_metrics(torch.cat(chunks), actual, model.transform.frequency)


def fit_forecast(checkpoint, train, validation, config, conditioned):
    model = ControlledForecast(checkpoint, conditioned)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['forecast_lr'])
    price_scale = config['cost_scale']
    branch_count = train['actions'].shape[1]
    generator = torch.Generator().manual_seed(config['seed'])
    best_state, best_score = None, float('inf')
    history = []
    for update in range(1, config['forecast_updates'] + 1):
        indices = torch.randint(len(train['levels']), (config['batch_size'],), generator=generator)
        columns = torch.randint(branch_count, (len(indices),), generator=generator)
        levels = train['levels'][indices]
        last = levels[:, -1]
        action = train['actions'][indices, columns]
        target = train['future'][indices, columns][:, [h - 1 for h in HORIZONS]]
        delta = model.forecast(levels, action)
        predicted = model.levels(last, delta)
        magnitude = model.transform.scale * last.new_tensor(HORIZONS)[None, :, None].sqrt()
        cv_loss = nn.functional.smooth_l1_loss((predicted[..., :41] - target[..., :41]) / magnitude[..., :41],
                                               torch.zeros_like(target[..., :41]))
        mv_loss = nn.functional.smooth_l1_loss(predicted[..., 41:52] / 2, target[..., 41:52] / 2)
        price_loss = nn.functional.smooth_l1_loss(cost(predicted[..., :41], model.transform.frequency) / price_scale,
                                                  cost(target[..., :41], model.transform.frequency) / price_scale)
        loss = cv_loss + mv_loss + price_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if update % 200 == 0 or update == config['forecast_updates']:
            model.eval()
            metrics = evaluate_forecast(model, validation)
            history.append(dict(update=update, loss=float(loss.detach()), metrics=metrics))
            score = sum(row['price_rmse'] for row in metrics.values())
            if score < best_score:
                best_state, best_score = copy.deepcopy(model.state_dict()), score
            model.train()
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return model, history


def select_forecast(checkpoint, train, validation, config):
    alternatives = {'original': ForecastFeatures(checkpoint)}
    learning = {}
    for name, conditioned in (('history', False), ('command', True)):
        alternatives[name], learning[name] = fit_forecast(checkpoint, train, validation, config, conditioned)
    metrics = {name: evaluate_forecast(model, validation) for name, model in alternatives.items()}
    selection = min(metrics, key=lambda name: sum(row['price_rmse'] for row in metrics[name].values()))
    return alternatives[selection], dict(selected=selection, validation=metrics, learning=learning)
