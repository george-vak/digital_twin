"""Train a Q head, freeze it, then optimize the actor on shared forecast features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import copy

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp_8.simulator import DockerSimulator
from exp_8.data import Environment
from exp_8.networks import Actor, Critic, ForecastFeatures, ControlledForecast, actor_update
from exp_8.collect import collect, signature
from exp_8.forecast import select_forecast
from exp_8.critic import fit_critic, inputs

COLUMNS = [f'CV[{i}]' for i in range(41)] + [f'MV[{i}]' for i in range(12)]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_config(config, transform):
    for key in ('episode_steps', 'warmup_steps', 'freeze_steps', 'hidden_size', 'critic_updates',
                'batch_size', 'updates_per_episode', 'episodes', 'regularization_window',
                'forecast_updates', 'intervention_period', 'target_streak', 'anchors',
                'anchor_spacing', 'branch_horizon'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('cost_scale', 'action_scale', 'regularization_scale',
                'critic_lr', 'actor_lr', 'gradient_clip', 'failure_penalty', 'tracking_tolerance',
                'residual_limit', 'forecast_lr', 'barrier_log_scale'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('tracking_weight', 'barrier_weight', 'action_weight', 'regularization_weight',
                'residual_weight', 'freeze_atol'):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not 0 < config['gamma'] < 1 or not 0 < config['barrier_epsilon'] < 1:
        raise ValueError('gamma and barrier_epsilon must be in (0, 1)')
    if not np.isfinite(config['target_cost']):
        raise ValueError('target_cost must be finite')
    if config['frequency_seconds'] != transform.frequency or config['warmup_steps'] < transform.window + 1:
        raise ValueError('Predictor frequency/history mismatch')
    if not 1 <= config['regularization_window'] <= transform.window + 1:
        raise ValueError('Invalid regularization window')
    if transform.window != 60 or config['branch_horizon'] < 10 or config['anchor_spacing'] < 2:
        raise ValueError('Expected 60-step history, branch horizon >= 10 and anchor spacing >= 2')
    if config['target_streak'] > config['episode_steps']:
        raise ValueError('Target streak exceeds the episode limit')
    groups = [config[key] for key in ('calibration_seeds', 'critic_validation_seeds',
                                      'validation_seeds', 'evaluation_seeds')]
    groups.append(list(range(config['train_seed_start'], config['train_seed_start'] + config['episodes'])))
    for group in groups:
        if not group or len(set(group)) != len(group) or any(type(seed) is not int or seed == 0 for seed in group):
            raise ValueError('Seed groups must be nonempty and unique')
    for index, group in enumerate(groups):
        if any(set(group) & set(other) for other in groups[index + 1:]):
            raise ValueError('Training, critic validation, validation and evaluation seeds must be disjoint')


def load_dataset(config, seeds):
    path = ROOT / config['dataset']
    if json.loads((path / 'protocol.json').read_text()) != signature(config):
        raise ValueError('Simulator or collection protocol has changed')
    parts = []
    for seed in seeds:
        with np.load(path / f'{seed}.npz', allow_pickle=False) as source:
            part = {key: torch.from_numpy(source[key]) for key in source.files}
        count = len(part['levels'])
        part['remaining'] = config['episode_steps'] - torch.arange(count) * config['anchor_spacing']
        parts.append(part)
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def save_rows(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def streak_metrics(costs, config):
    errors = np.asarray(costs) - config['target_cost']
    inside = np.abs(errors) <= config['tracking_tolerance']
    edges = np.flatnonzero(np.diff(np.r_[False, inside, False]))
    intervals = list(zip(edges[::2], edges[1::2]))
    longest = max((end - start for start, end in intervals), default=0)
    length = min(longest, config['target_streak'])
    if length == 0:
        return dict(longest_streak=0, streak_success=False, streak_rmse=None, streak_std=None, streak_start=None)
    start = min((i for begin, end in intervals for i in range(begin, end - length + 1)),
                key=lambda i: np.mean(errors[i:i + length] ** 2))
    window = errors[start:start + length]
    return dict(longest_streak=int(longest), streak_success=bool(longest >= config['target_streak']),
                streak_rmse=float(np.sqrt(np.mean(window ** 2))), streak_std=float(window.std()),
                streak_start=int(start + 1))


@torch.no_grad()
def run_episode(backend, encoder, actor, config, seed, directory, policy, label_period=0):
    if policy not in ('actor_only', 'actor', 'internal'):
        raise ValueError(f'Unknown control policy {policy}')
    if label_period and policy != 'actor_only':
        raise ValueError('Expert labels can only be requested on actor-only training trajectories')
    env = Environment(backend, config, encoder.transform.window)
    env.reset(seed)
    if directory is not None:
        write_json(directory / f'warmup_{seed}_bounds.json', env.bounds)
        np.savetxt(directory / f'warmup_{seed}.csv', env.warmup[-1:], delimiter=',',
                   header=','.join(COLUMNS), comments='')
    initial_violation = env.cv_violation
    lower = torch.tensor(env.bounds['lower'], dtype=torch.float32)[None]
    upper = torch.tensor(env.bounds['upper'], dtype=torch.float32)[None]
    rows, samples = [], []

    def record(transition, intervention, residual, previous_mv):
        following, price, terminated, truncated = transition
        observed = following[-1]
        rows.append(dict(seed=seed, step=env.steps, cost=price, terminated=terminated,
                         truncated=truncated, cv_violation=env.cv_violation,
                         safe_step=not bool(env.cv_violation), intervention=intervention,
                         residual_mv=residual, mean_abs_delta_mv=float(np.abs(observed[41:52] - previous_mv).mean()),
                         **dict(zip(COLUMNS, observed.tolist()))))

    while not env.done:
        remaining = config['episode_steps'] - env.steps
        if policy == 'actor_only':
            levels = torch.tensor(env.levels, dtype=torch.float32)[None]
            clock = torch.tensor([remaining])
            anchor = torch.tensor(env.warmup[-1:, 41:52], dtype=torch.float32)
            inputs = actor.observe(encoder, levels, lower, upper, clock, config, anchor)
            command = actor(inputs, levels[:, -1, 41:52])[0].numpy()
            if label_period and env.steps % label_period == 0:
                teacher = torch.tensor(backend.nominal(), dtype=torch.float32)[None]
                samples.append(dict(inputs=inputs[0], last=levels[0, -1, 41:52], teacher=teacher[0]))
            previous = env.levels[-1, 41:52].copy()
            record(env.step(np.append(command, env.bounds['fixed_mv11'])), True, 0., previous)
            count = 0
        elif policy != 'internal':
            nominal = backend.nominal()
            nominal[11] = env.bounds['fixed_mv11']
            levels = torch.tensor(env.levels, dtype=torch.float32)[None]
            nominal_input = torch.tensor(nominal, dtype=torch.float32)[None]
            features, _ = encoder(levels, lower, upper, torch.tensor([remaining]), config, nominal_input)
            last = levels[:, -1]
            reference = levels[:, -config['regularization_window']:, 41:52].mean(dim=1)
            command = actor(features, nominal_input, config)[0].numpy()
            residual = float(np.abs(command - nominal[:11]).mean())
            previous = env.levels[-1, 41:52].copy()
            record(env.step(np.append(command, nominal[11])), True, residual, previous)
            samples.append(dict(features=features[0], last=last[0], reference=reference[0], nominal=nominal_input[0]))
            count = min(config['intervention_period'] - 1, remaining - 1)
        else:
            count = min(30, remaining)
        if count and not env.done:
            observations, prices = backend.internal(count)
            for row, price in zip(observations, prices):
                previous = env.levels[-1, 41:52].copy()
                record(env.consume(row, price), False, 0.0, previous)
                if env.done:
                    break
    costs = np.array([row['cost'] for row in rows])
    summary = dict(seed=seed, steps=len(rows), completed=rows[-1]['truncated'],
                   termination_reason=env.termination_reason, initial_cv_violation=initial_violation,
                   mean_cost=float(costs.mean()), cost_mae=float(np.abs(costs - config['target_cost']).mean()),
                   cost_std=float(costs.std()), safe_steps=sum(row['safe_step'] for row in rows),
                   target_steps=int((np.abs(costs - config['target_cost']) <= config['tracking_tolerance']).sum()),
                   mean_abs_delta_mv=float(np.mean([row['mean_abs_delta_mv'] for row in rows])),
                   **streak_metrics(costs, config))
    return rows, summary, samples


def aggregate(episodes, config):
    errors = [row['streak_rmse'] for row in episodes if row['streak_rmse'] is not None]
    successes = [row for row in episodes if row['streak_success']]
    achieved = [row['streak_rmse'] for row in successes]
    return dict(episodes=episodes, mean_steps=float(np.mean([row['steps'] for row in episodes])),
                max_steps=max(row['steps'] for row in episodes),
                completed_episodes=sum(row['completed'] for row in episodes),
                episodes_at_least_200=sum(row['steps'] >= config['target_streak'] for row in episodes),
                successful_episodes=len(successes),
                median_streak=float(np.median([row['longest_streak'] for row in episodes])),
                mean_streak=float(np.mean([row['longest_streak'] for row in episodes])),
                max_streak=max(row['longest_streak'] for row in episodes),
                mean_streak_rmse=float(np.mean(achieved or errors)) if errors else None,
                mean_abs_delta_mv=float(np.mean([row['mean_abs_delta_mv'] for row in episodes])),
                cost_mae=float(np.average([row['cost_mae'] for row in episodes],
                                         weights=[row['steps'] for row in episodes])))


def score(summary, config):
    del config
    error = summary['mean_streak_rmse']
    return (-summary['successful_episodes'], -summary['median_streak'], -summary['mean_streak'],
            error if error is not None else float('inf'), summary['cost_mae'])


def evaluate(backend, encoder, actor, config, seeds, directory, policy='actor'):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'config.json', config)
    results, traces = [], []
    for seed in seeds:
        rows, summary, _ = run_episode(backend, encoder, actor, config, seed, directory, policy)
        results.append(summary)
        traces.extend(rows)
    summary = aggregate(results, config)
    save_rows(directory / 'steps.csv', traces)
    write_json(directory / 'summary.json', summary)
    print(json.dumps(dict(phase=directory.name, **compact(summary)), ensure_ascii=False), flush=True)
    return summary


def compact(summary):
    return {key: value for key, value in summary.items() if key != 'episodes'}


def optimize_actor(backend, encoder, critic, initial_replay, config, directory):
    directory.mkdir()
    torch.manual_seed(config['seed'])
    actor = Actor(config['hidden_size'])
    optimizer = torch.optim.AdamW(actor.parameters(), lr=config['actor_lr'])
    replay = initial_replay.copy()
    best = None
    trace, all_rows, train_episodes, curve = [], [], [], []
    for epoch in range(1, config['episodes'] + 1):
        for _ in range(config['updates_per_episode']):
            indices = torch.randint(len(replay['last']), (config['batch_size'],))
            batch = {key: replay[key][indices] for key in ('features', 'last', 'reference', 'nominal')}
            metrics = actor_update(actor, critic, optimizer, config=config, **batch)
            trace.append(dict(update=len(trace) + 1, epoch=epoch, **metrics))
        rows, summary, samples = run_episode(backend, encoder, actor, config,
                                            config['train_seed_start'] + epoch - 1, directory, 'actor')
        all_rows.extend(rows)
        train_episodes.append(summary)
        for key in ('features', 'last', 'reference', 'nominal'):
            replay[key] = torch.cat((replay[key], torch.stack([sample[key] for sample in samples])))
        save_rows(directory / 'actor_learning.csv', trace)
        result = evaluate(backend, encoder, actor, config, config['validation_seeds'],
                          directory / f'validation_{epoch:03d}')
        curve.append(dict(epoch=epoch, **compact(result)))
        save_rows(directory / 'validation_curve.csv', curve)
        if best is None or score(result, config) < score(best[1], config):
            best = epoch, result, copy.deepcopy(actor.state_dict())
    write_json(directory / 'config.json', config)
    write_json(directory / 'summary.json', aggregate(train_episodes, config))
    save_rows(directory / 'steps.csv', all_rows)
    return best


def restore_forecast(source, saved):
    if saved['forecast'] == 'original':
        encoder = ForecastFeatures(source)
    else:
        encoder = ControlledForecast(source, saved['forecast'] == 'command')
    encoder.load_state_dict(saved['encoder'])
    encoder.eval().requires_grad_(False)
    if state_hash(encoder) != saved['hashes']['predictor']:
        raise ValueError('Predictor weights differ from the trained controller')
    return encoder


def train(config, output, model_directory=None):
    torch.manual_seed(config['seed'])
    source = torch.load(ROOT / config['checkpoint'], map_location='cpu', weights_only=False)
    original = ForecastFeatures(source)
    validate_config(config, original.transform)
    train_data = load_dataset(config, config['calibration_seeds'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    if model_directory is None:
        validation_data = load_dataset(config, config['critic_validation_seeds'])
        encoder, forecast = select_forecast(source, train_data, validation_data, config)
        _, _, original_q = fit_critic(original, train_data, validation_data, config)
        critic, replay, improved_q = fit_critic(encoder, train_data, validation_data, config)
    else:
        saved = torch.load(model_directory / 'controller.pt', map_location='cpu', weights_only=False)
        shared_keys = ('checkpoint', 'episode_steps', 'frequency_seconds', 'hidden_size', 'cost_scale',
                       'target_cost', 'tracking_weight', 'gamma', 'failure_penalty', 'barrier_weight',
                       'barrier_log_scale', 'barrier_epsilon', 'residual_limit', 'branch_horizon',
                       'calibration_seeds', 'critic_validation_seeds', 'dataset')
        if any(config[key] != saved['configuration'][key] for key in shared_keys):
            raise ValueError('Forecast and critic settings differ from the saved controller')
        encoder = restore_forecast(source, saved)
        critic = Critic(config['hidden_size'])
        critic.load_state_dict(saved['critic'])
        critic.eval().requires_grad_(False)
        if state_hash(critic) != saved['hashes']['critic']:
            raise ValueError('Critic weights differ from the saved controller')
        replay = inputs(encoder, train_data, config)
        forecast = json.loads((model_directory / 'forecast.json').read_text())
        diagnostics = json.loads((model_directory / 'critic.json').read_text())
        original_q, improved_q = diagnostics['original_features'], diagnostics['updated_features']
    torch.save(encoder.state_dict(), output / 'forecast.pt')
    write_json(output / 'forecast.json', forecast)
    print(json.dumps(dict(stage='forecast', selected=forecast['selected'], validation=forecast['validation'])), flush=True)
    write_json(output / 'critic.json', dict(original_features=original_q, updated_features=improved_q))
    frozen = dict(predictor=state_hash(encoder), critic=state_hash(critic))
    print(json.dumps(dict(stage='critic', **{key: value for key, value in improved_q.items() if key != 'history'})),
          flush=True)

    backend = DockerSimulator(config['docker_image'])
    try:
        initial = Actor(config['hidden_size']).eval()
        baseline = evaluate(backend, encoder, initial, config, config['validation_seeds'],
                            output / 'baseline_validation', 'internal')
        options = []
        for index, option in enumerate(config['actor_candidates']):
            settings = {**config, **option}
            validate_config(settings, original.transform)
            if not 0 < settings.get('correction_gain', 1.0) <= 1:
                raise ValueError('correction_gain must be between 0 and 1, inclusive of 1')
            epoch, result, weights = optimize_actor(backend, encoder, critic, replay, settings,
                                                   output / f'candidate_{index:02d}')
            options.append((index, epoch, result, weights, settings))
        index, epoch, result, weights, settings = min(options, key=lambda item: score(item[2], config))
        if frozen != dict(predictor=state_hash(encoder), critic=state_hash(critic)):
            raise RuntimeError('Frozen network weights changed during actor training')
        write_json(output / 'selection.json',
                   dict(index=index, epoch=epoch, settings=settings, frozen_hashes=frozen,
                        candidates=[dict(index=i, epoch=e, settings=config['actor_candidates'][i], **compact(r))
                                    for i, e, r, _, _ in options],
                        baseline=compact(baseline), selected=compact(result)))
        torch.save(dict(actor=weights, critic=critic.state_dict(), encoder=encoder.state_dict(),
                        configuration=settings, forecast=forecast['selected'], hashes=frozen), output / 'controller.pt')
        actor = Actor(config['hidden_size'])
        actor.load_state_dict(weights)
        actor.eval()
        evaluate(backend, encoder, actor, settings, config['validation_seeds'], output / 'validation')
        evaluation = evaluate(backend, encoder, actor, settings, config['evaluation_seeds'], output / 'evaluation')
        reference = evaluate(backend, encoder, initial, settings, config['evaluation_seeds'],
                             output / 'baseline_evaluation', 'internal')
        nominal = evaluate(backend, encoder, initial, settings, config['evaluation_seeds'],
                           output / 'nominal_evaluation')
        write_json(output / 'result.json', dict(actor=evaluation, internal=reference, nominal=nominal))
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='rebuilt')
    parser.add_argument('--phase', choices=('prepare', 'train', 'evaluation'), default='train')
    parser.add_argument('--config', type=Path, default=ROOT / 'exp_8/config.json')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--reuse-models', type=Path, help='Reuse a run directory with a frozen GRU and Q critic')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..') or args.threads < 1:
        parser.error('Invalid run name or number of threads')
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text())
    output = ROOT / 'exp_8/runs' / args.run
    if args.phase == 'prepare':
        collect(config)
    elif args.phase == 'train':
        train(config, output, args.reuse_models)
    else:
        saved = torch.load(output / 'controller.pt', map_location='cpu', weights_only=False)
        config = saved['configuration']
        source = torch.load(ROOT / config['checkpoint'], map_location='cpu', weights_only=False)
        encoder = restore_forecast(source, saved)
        actor = Actor(config['hidden_size'])
        actor.load_state_dict(saved['actor'])
        actor.eval()
        backend = DockerSimulator(config['docker_image'])
        try:
            evaluate(backend, encoder, actor, config, config['evaluation_seeds'], output / 'repeat_evaluation')
        finally:
            backend.close()


if __name__ == '__main__':
    main()
