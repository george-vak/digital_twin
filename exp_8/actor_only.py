"""Learn a policy that issues every process command after simulator warmup."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from exp_8.networks import FEATURES
from exp_8.run import (ROOT, aggregate, load_dataset, restore_forecast, run_episode,
                       save_rows, state_hash, write_json, validate_config)
from exp_8.simulator import DockerSimulator


class FullActor(nn.Module):
    """Imitate the expert with bounded commands around the warmup MV."""
    def __init__(self, hidden):
        super().__init__()
        self.imitation = nn.Sequential(nn.Linear(FEATURES + 115, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 11))

    @staticmethod
    def observe(encoder, levels, lower, upper, remaining, config, anchor):
        features, _ = encoder(levels, lower, upper, remaining, config)
        current = levels[:, -1, :52]
        width = (upper - lower).clamp_min(.01)
        recent = (current - levels[:, -4, :52]) / width
        slower = (current - levels[:, -16, :52]) / width
        inputs = torch.cat((features, recent, slower), dim=1).clamp(-15, 15)
        return torch.cat((inputs, anchor.expand(len(levels), -1)), dim=1)

    def forward(self, features, last):
        anchor = features[:, -11:]
        network_inputs = torch.cat((features[:, :-11], (last - anchor) / 5.), dim=1)
        limit = last.new_tensor([3, 3, 15, 3, 2, 10, 4, 4, 6, 15, 4])
        return (anchor + limit * self.imitation(network_inputs).tanh()).clamp(0, 100)


@torch.no_grad()
def expert_dataset(encoder, config):
    records = []
    for seed in config['expert_seeds']:
        path = ROOT / config['expert_data'] / f'{seed}.npz'
        with np.load(path, allow_pickle=False) as source:
            values = torch.tensor(source['levels'])
            lower = torch.tensor(source['lower'], dtype=torch.float32)[None]
            upper = torch.tensor(source['upper'], dtype=torch.float32)[None]
        levels = values.unfold(0, 61, 1).transpose(1, 2)[:-1]
        targets = values[61:, 41:]
        for start in range(0, len(targets), 64):
            stop = min(start + 64, len(targets))
            history, teacher = levels[start:stop], targets[start:stop]
            last = history[:, -1, 41:52]
            remaining = config['episode_steps'] - torch.arange(start, stop)
            inputs = FullActor.observe(encoder, history, lower, upper, remaining, config, values[60:61, 41:52])
            records.append(dict(inputs=inputs, last=last, teacher=teacher))
    return {key: torch.cat([batch[key] for batch in records]) for key in records[0]}


@torch.no_grad()
def branch_labels(encoder, config):
    labels = []
    for seed in config['calibration_seeds']:
        data = load_dataset(config, [seed])
        levels, nominal = data['levels'], data['nominal']
        anchor = levels[:1, -1, 41:52]
        lower, upper, remaining = (data[key] for key in ('lower', 'upper', 'remaining'))
        features = FullActor.observe(encoder, levels, lower, upper, remaining, config, anchor)
        labels.append(dict(inputs=features, last=levels[:, -1, 41:52], teacher=nominal))
    return {key: torch.cat([batch[key] for batch in labels]) for key in labels[0]}


def fit_actor(actor, replay, config, updates):
    optimizer = torch.optim.AdamW(actor.parameters(), lr=config['imitation_lr'])
    for _ in range(updates):
        indices = torch.randint(len(replay['last']), (config['imitation_batch'],))
        features, last, teacher = (replay[key][indices] for key in ('inputs', 'last', 'teacher'))
        command = actor(features, last)
        imitation = nn.functional.smooth_l1_loss(command, teacher[:, :11], beta=.35)
        loss = imitation + config['movement_weight'] * (command - last).abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


def summarize(episodes, traces, config):
    summary = aggregate(episodes, config)
    summary.update(mean_cost=float(np.mean([row['cost'] for row in traces])),
                   at_least_200_cost=(float(np.mean([row['mean_cost'] for row in episodes if row['steps'] >= 200]))
                                      if any(row['steps'] >= 200 for row in episodes) else None))
    return summary


def score(summary):
    # Stable completion first, then the time-weighted production cost.
    shortfall = sum(max(0, 200 - episode['steps']) for episode in summary['episodes'])
    return (-summary['episodes_at_least_200'], -summary['completed_episodes'], shortfall, summary['mean_cost'])


def sample(backend, encoder, actor, config, seeds, output, labels=0, policy='actor_only'):
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'config.json', config)
    episodes, traces, data = [], [], []
    for seed in seeds:
        rows, summary, samples = run_episode(backend, encoder, actor, config, seed, output, policy, labels)
        if policy == 'actor_only' and not all(row['intervention'] for row in rows):
            raise RuntimeError('Actor-only rollout contains an internal-control step')
        traces.extend(rows)
        episodes.append(summary)
        data.extend(samples)
    summary = summarize(episodes, traces, config)
    if output is not None:
        save_rows(output / 'steps.csv', traces)
        write_json(output / 'summary.json', summary)
        print(json.dumps(dict(stage=output.name, **{k: v for k, v in summary.items() if k != 'episodes'})), flush=True)
    return summary, data


def train(config, destination, backbone):
    snapshot = torch.load(backbone / 'controller.pt', weights_only=False, map_location='cpu')
    source = torch.load(ROOT / config['checkpoint'], weights_only=False, map_location='cpu')
    encoder = restore_forecast(source, snapshot)
    validate_config(config, encoder.transform)
    for key in ('checkpoint', 'episode_steps', 'frequency_seconds', 'residual_limit', 'target_cost'):
        if config[key] != snapshot['configuration'][key]:
            raise ValueError(f'Actor-only experiment must match the backbone on {key}')
    groups = [set(config['expert_seeds']), *(set(seeds) for seeds in config['dagger_seeds']),
              set(config['actor_validation_seeds']), set(config['evaluation_seeds']),
              set(config['calibration_seeds']), set(config['critic_validation_seeds'])]
    if sum(map(len, groups)) != len(set().union(*groups)):
        raise ValueError('Training, validation and evaluation seed groups overlap')
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'config.json', config)
    csv_rows = []
    frozen = state_hash(encoder)
    replay = expert_dataset(encoder, config)
    off_policy = branch_labels(encoder, config)
    for key in replay:
        replay[key] = torch.cat((replay[key], off_policy[key].repeat((6, 1))))
    torch.manual_seed(config['seed'])
    actor = FullActor(config['actor_hidden'])
    fit_actor(actor, replay, config, config['imitation_updates'])
    backend = DockerSimulator(config['docker_image'])
    try:
        def candidate(name, model):
            summary, _ = sample(backend, encoder, model, config, config['actor_validation_seeds'],
                                destination / name)
            csv_rows.append(dict(candidate=name, **{key: summary[key] for key in
                                                    ('episodes_at_least_200', 'completed_episodes', 'mean_steps', 'mean_cost')}))
            save_rows(destination / 'comparison.csv', csv_rows)
            torch.save(dict(actor=model.state_dict()), destination / name / 'actor.pt')
            return copy.deepcopy(model), summary, name

        candidates = [candidate('imitation', actor)]
        for index, seeds in enumerate(config['dagger_seeds'], 1):
            _, labelled = sample(backend, encoder, actor, config, seeds, destination / f'collection_{index}', labels=2)
            labelled = {key: torch.stack([row[key] for row in labelled]) for key in replay}
            replay_weight = config.get('expert_replay_weight', len(replay['last']) // 3)
            repeats = max(1, replay_weight // len(labelled['last']))
            for key in replay:
                replay[key] = torch.cat((replay[key], torch.cat([labelled[key]] * repeats)))
            fit_actor(actor, replay, config, config['dagger_updates'])
            candidates.append(candidate(f'imitation_{index}', actor))

        selected, validation, name = min(candidates, key=lambda entry: score(entry[1]))
        if frozen != state_hash(encoder):
            raise RuntimeError('Frozen forecast weights changed')
        write_json(destination / 'selection.json', dict(candidate=name, encoder_sha256=frozen,
                   validation={key: value for key, value in validation.items() if key != 'episodes'}))
        torch.save(dict(actor=selected.state_dict(), config=config, forecast=snapshot['forecast'],
                        encoder=encoder.state_dict(), hashes=dict(predictor=frozen)),
                   destination / 'controller.pt')
        for phase, policy in [('evaluation', 'actor_only'), ('baseline_evaluation', 'internal')]:
            sample(backend, encoder, selected, config, config['evaluation_seeds'], destination / phase, policy=policy)
    finally:
        backend.close()


def restore_actor(saved):
    config = saved['config']
    if saved.get('gain', 0) or not config.get('anchored', True):
        raise ValueError('Expected a checkpoint of the anchored imitation policy')
    actor = FullActor(config['actor_hidden'])
    weights = {key: value for key, value in saved['actor'].items() if not key.startswith('refinement.')}
    actor.load_state_dict(weights)
    return actor.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp_8/actor_anchored.json')
    parser.add_argument('--run', default='actor_anchored')
    parser.add_argument('--backbone', type=Path, default=ROOT / 'exp_8/runs/rebuilt')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--phase', choices=('train', 'evaluation'), default='train')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..') or args.threads < 1:
        parser.error('Invalid run name or number of threads')
    torch.set_num_threads(args.threads)
    directory = ROOT / 'exp_8/runs' / args.run
    if args.phase == 'train':
        train(json.loads(args.config.read_text()), directory, args.backbone)
    else:
        saved = torch.load(directory / 'controller.pt', map_location='cpu', weights_only=False)
        config = saved['config']
        source = torch.load(ROOT / config['checkpoint'], map_location='cpu', weights_only=False)
        encoder = restore_forecast(source, saved)
        actor = restore_actor(saved)
        backend = DockerSimulator(config['docker_image'])
        try:
            sample(backend, encoder, actor, config, config['evaluation_seeds'], directory / 'repeat_evaluation')
        finally:
            backend.close()


if __name__ == '__main__':
    main()
