"""Sample paired responses to MV commands from exact simulator clones."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from exp_8.safety import fit_bounds
from exp_8.simulator import DockerSimulator, ROOT

CONFIG = ROOT / 'exp_8/config.json'


def signature(config):
    keys = ('warmup_steps', 'frequency_seconds', 'branch_horizon', 'anchors',
            'anchor_spacing', 'residual_limit', 'calibration_seeds', 'critic_validation_seeds')
    return dict(settings={key: config[key] for key in keys},
                source={key: hashlib.sha256((ROOT / key).read_bytes()).hexdigest()
                        for key in ('libtemain.so', 'temain.py', 'exp_8/simulator.py', 'exp_8/collect.py')})


def collect(config):
    output = ROOT / config['dataset']
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / 'protocol.json'
    protocol = signature(config)
    if metadata.exists() and json.loads(metadata.read_text()) != protocol:
        raise ValueError('Collection settings or simulator changed. Choose a new dataset path.')
    metadata.write_text(json.dumps(protocol, indent=2) + '\n')
    backend = DockerSimulator(config['docker_image'])
    try:
        for seed in config['calibration_seeds'] + config['critic_validation_seeds']:
            path = output / f'{seed}.npz'
            if path.exists():
                continue
            warmup = backend.reset(seed=seed, warmup_steps=config['warmup_steps'],
                                   window=config['warmup_steps'] - 1,
                                   frequency_seconds=config['frequency_seconds'])
            bounds = fit_bounds(warmup)
            levels = warmup[-61:]
            rng = np.random.default_rng(seed)
            groups = []
            for index in range(config['anchors']):
                nominal = backend.nominal()
                nominal[11] = bounds['fixed_mv11']
                directions = rng.uniform(-1, 1, (3, 11))
                # Include one axis perturbation and two simultaneous perturbations.
                channel = (index + seed) % 11
                directions[0] = np.eye(11)[channel]
                changes = config['residual_limit'] * np.concatenate((np.zeros((1, 11)), directions, -directions))
                actions = np.repeat(nominal[None], len(changes), axis=0)
                actions[:, :11] = np.clip(actions[:, :11] + changes, 0, 100)
                responses = backend.branches(actions.tolist(), config['branch_horizon'])
                observations = np.asarray([row for row, _ in responses])
                prices = np.asarray([price for _, price in responses])
                groups.append(dict(levels=levels, nominal=nominal, actions=actions, future=observations,
                                   prices=prices, lower=bounds['lower'], upper=bounds['upper']))
                # Alternate nominal internal control and a sampled intervention so the
                # histories cover both unperturbed and perturbed operating states.
                if index % 2 == 0:
                    following, _ = backend.step(actions[rng.integers(1, len(actions))])
                    tail, _ = backend.internal(config['anchor_spacing'] - 1)
                    new_levels = np.vstack((following, tail))
                else:
                    tail, _ = backend.internal(config['anchor_spacing'])
                    new_levels = np.asarray(tail)
                levels = np.concatenate((levels, new_levels))[-61:]
                if np.all(np.abs(np.diff(levels[-4:, :41], axis=0)) <= config['freeze_atol']):
                    break
            data = {key: np.asarray([group[key] for group in groups], dtype=np.float32) for key in groups[0]}
            if any(not np.isfinite(values).all() for values in data.values()):
                raise FloatingPointError(f'Non-finite data for seed {seed}')
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **data)
            temporary.replace(path)
            print(json.dumps(dict(seed=seed, anchors=len(groups), branches=len(groups) * 7)), flush=True)
    finally:
        backend.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    collect(json.loads(args.config.read_text()))
