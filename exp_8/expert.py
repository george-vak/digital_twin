"""Collect expert trajectories for imitation of the built-in TE controller."""
import argparse
import json
from pathlib import Path

import numpy as np

from exp_8.safety import fit_bounds
from exp_8.simulator import ROOT, DockerSimulator


def collect(config, output):
    output.mkdir(parents=True, exist_ok=True)
    backend = DockerSimulator(config['docker_image'])
    try:
        for seed in config['expert_seeds']:
            path = output / f'{seed}.npz'
            if path.exists():
                continue
            warmup = backend.reset(seed=seed, warmup_steps=config['warmup_steps'],
                                   window=config['warmup_steps'] - 1,
                                   frequency_seconds=config['frequency_seconds'])
            levels, _ = backend.internal(config['episode_steps'])
            bounds = fit_bounds(warmup)
            levels = np.asarray(levels)
            if not np.isfinite(levels).all():
                raise ValueError(f'Non-finite expert trajectory {seed}')
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, levels=np.vstack((warmup[-61:], levels)).astype(np.float32),
                                lower=bounds['lower'], upper=bounds['upper'])
            temporary.replace(path)
            print(f'Expert seed {seed}, {len(levels)} steps', flush=True)
    finally:
        backend.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp_8/actor_anchored.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    collect(config, ROOT / config['expert_data'])
