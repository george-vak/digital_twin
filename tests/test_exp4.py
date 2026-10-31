import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from exp_3.networks import actor_update as baseline_update
from exp_4.networks import actor_update, validate_regularization
from exp_4.run import Replay


class Transform:
    frequency = 60

    def history(self, levels):
        return levels[:, 1:] - levels[:, :-1]

    def cv_target(self, last, following):
        return following[:, :41] - last[:, :41]

    def cv_levels(self, last, delta):
        return last[:, :41] + delta


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Parameter(torch.arange(12, dtype=torch.float32) + 40)

    def forward(self, history, last, transform):
        return self.action.expand(last.shape[0], -1)


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(0.01))

    def forward(self, history, last, action, transform):
        return action.mean(-1, keepdim=True).expand(-1, 41) * self.factor


class RegularizationTests(unittest.TestCase):
    def test_replay_reference_is_per_channel_past_tail(self):
        replay = Replay(10, 42, 2)
        levels = np.zeros((4, 53), dtype=np.float32)
        levels[:, 41:] = np.arange(4)[:, None] * 10 + np.arange(12)
        replay.add(levels, np.full(12, 99), np.full(53, 999))
        *_, reference = replay.sample(1, Transform())
        torch.testing.assert_close(reference[0], torch.arange(12, dtype=torch.float32) + 25)

    def test_zero_weight_matches_baseline(self):
        actor = Actor()
        baseline = copy.deepcopy(actor)
        model = Dynamics()
        other_model = copy.deepcopy(model)
        last = torch.ones(2, 53)
        history = torch.zeros(2, 3, 53)
        opt = torch.optim.SGD(actor.parameters(), lr=0.1)
        other_opt = torch.optim.SGD(baseline.parameters(), lr=0.1)
        result = actor_update(actor, model, Transform(), opt, history, last, torch.zeros(2, 12), 0, 100)
        expected = baseline_update(baseline, other_model, Transform(), other_opt, history, last)
        self.assertEqual(result['actor_loss'], expected)
        torch.testing.assert_close(actor.action, baseline.action)
        self.assertIsNone(model.factor.grad)
        self.assertTrue(model.factor.requires_grad)

    def test_penalty_pulls_channels_towards_their_own_reference(self):
        actor, model = Actor(), Dynamics()
        with torch.no_grad():
            model.factor.zero_()  # Isolate the regularization gradient.
        before = actor.action.detach().clone()
        reference = before[None].clone()
        reference[0, 0] += 10
        reference[0, 1] -= 10
        reference.requires_grad_()
        result = actor_update(actor, model, Transform(), torch.optim.SGD(actor.parameters(), lr=1),
                              torch.zeros(1, 3, 53), torch.zeros(1, 53), reference, 2, 100)
        self.assertGreater(actor.action[0], before[0])
        self.assertLess(actor.action[1], before[1])
        torch.testing.assert_close(actor.action[2:], before[2:])
        self.assertAlmostEqual(result['actor_penalty'], 0.02 / 12)
        self.assertAlmostEqual(result['actor_loss'], result['actor_cost'] + result['actor_regularization'])
        self.assertIsNone(reference.grad)

    def test_invalid_configuration(self):
        config = dict(regularization_window=60, regularization_weight=1., regularization_scale=100.)
        validate_regularization(config, 61)
        for key, value in [('regularization_window', 62), ('regularization_window', 0),
                           ('regularization_weight', -1), ('regularization_weight', float('nan')),
                           ('regularization_scale', 0)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_regularization({**config, key: value}, 61)

    def test_only_requested_plots(self):
        from exp_4.plot import plot_run
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'config.json').write_text(json.dumps(dict(episode_steps=2, frequency_seconds=60)))
            rows = []
            for step in (1, 2):
                row = dict(phase='train', episode=0, step=step, cost=3., actor_loss=4., terminated=False)
                row.update({f'CV[{j}]': float(j) for j in range(41)})
                row.update({f'MV[{j}]': float(j) for j in range(12)})
                rows.append(row)
            pd.DataFrame(rows).to_csv(run / 'steps.csv', index=False)
            paths = plot_run(run)
            self.assertEqual({p.name for p in paths},
                             {'01_training.png', '03_cv_episode_000.png', '04_mv_episode_000.png'})
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))


if __name__ == '__main__':
    unittest.main()
