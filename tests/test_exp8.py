import unittest
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch

from exp_8.data import Environment
from exp_8.networks import Actor, Critic, ControlledForecast, ForecastFeatures, actor_update, FEATURES
from exp_8.critic import returns
from exp_8.safety import barrier_per_sample, relaxed_log
from exp_8.run import run_episode, state_hash, streak_metrics, validate_config

ROOT = Path(__file__).resolve().parents[1]


class Experiment8Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'exp_8/config.json').read_text())

    def test_actor_only_issues_every_command_without_querying_teacher(self):
        class Backend:
            commands = []

            def reset(self, **kwargs):
                self.commands = []
                return np.full((kwargs['warmup_steps'], 53), 50.)

            def step(self, command):
                self.commands.append(command.copy())
                row = np.full(53, 50.)
                row[0] += len(self.commands)
                row[41:] = command
                return row, 160.

        class Policy:
            @staticmethod
            def observe(encoder, levels, lower, upper, clock, config, anchor):
                return clock

            def __call__(self, clock, previous):
                return previous + clock[:, None]

        backend = Backend()
        config = dict(self.config, episode_steps=5, target_streak=5)
        encoder = SimpleNamespace(transform=SimpleNamespace(window=60))
        rows, outcome, labels = run_episode(backend, encoder, Policy(), config, 1, None, 'actor_only')
        self.assertEqual(outcome['steps'], 5)
        self.assertTrue(all(row['intervention'] for row in rows))
        self.assertEqual(labels, [])
        np.testing.assert_array_equal(np.stack(backend.commands)[:, 0], [55, 59, 62, 64, 65])
        np.testing.assert_array_equal(np.stack(backend.commands)[:, 11], 50)

    def test_violations_do_not_terminate_even_at_reset(self):
        class Backend:
            def reset(self, **kwargs):
                values = np.tile(np.arange(1000)[:, None], (1, 53)).astype(float)
                values[:, 52] = 50
                return values
            def step(self, action):
                self.value = getattr(self, 'value', 1000) + 1
                row = np.full(53, self.value, dtype=float)
                row[41:] = action
                return row, 160.0
        config = dict(self.config, episode_steps=1000)
        env = Environment(Backend(), config, 60)
        env.reset(1)
        self.assertTrue(env.cv_violation)
        self.assertFalse(env.done)
        for step in range(1000):
            _, _, terminated, truncated = env.step(np.full(12, 50.0))
            self.assertTrue(env.cv_violation)
            self.assertFalse(terminated)
            self.assertEqual(truncated, step == 999)
        with self.assertRaises(ValueError):
            env.step(np.full(12, 50.0))

    def test_frozen_process_still_terminates(self):
        class Backend:
            def step(self, action):
                return np.full(53, 50.0), 160.0
        env = Environment(Backend(), self.config, 2)
        env.levels = np.full((3, 53), 50.0)
        env.bounds = dict(lower=[0.0] * 52, upper=[100.0] * 52, fixed_mv11=50.0)
        env.steps = env.frozen = 0
        env.done = False
        for _ in range(3):
            _, _, terminated, truncated = env.step(np.full(12, 50.0))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env.termination_reason, 'simulator_frozen')

    def test_barrier_finite_symmetric_and_pushes_inward(self):
        values = torch.tensor([[-0.1], [0.1], [0.5], [0.9], [1.1]], requires_grad=True)
        result = barrier_per_sample(values, torch.zeros_like(values), torch.ones_like(values))
        self.assertTrue(torch.isfinite(result).all())
        self.assertAlmostEqual(result[2].item(), 0.0)
        torch.testing.assert_close(result, result.flip(0))
        result.sum().backward()
        self.assertLess(values.grad[0].item(), 0)
        self.assertGreater(values.grad[-1].item(), 0)
        self.assertAlmostEqual(values.grad[2].item(), 0.0)
        point = barrier_per_sample(torch.tensor([[0.001]]), torch.zeros(1, 1), torch.zeros(1, 1))
        self.assertAlmostEqual(point.item(), 1.0)

    def test_relaxed_log_matches_two_derivatives_at_join(self):
        gradients = []
        for value in (0.1 - 1e-8, 0.1 + 1e-8):
            x = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            y = relaxed_log(x, 0.1)
            first, = torch.autograd.grad(y, x, create_graph=True)
            second, = torch.autograd.grad(first, x)
            gradients.append((y.item(), first.item(), second.item()))
        np.testing.assert_allclose(*gradients, rtol=1e-6, atol=1e-6)

    def test_shared_forecast_really_outputs_all_53_variables(self):
        source = torch.load(ROOT / self.config['checkpoint'], weights_only=False, map_location='cpu')
        encoder = ForecastFeatures(source)
        levels = torch.zeros(2, encoder.transform.window + 1, 53)
        features, forecast = encoder(levels, torch.zeros(2, 52), torch.ones(2, 52), torch.ones(2), self.config)
        self.assertEqual(features.shape, (2, FEATURES))
        self.assertEqual(forecast.shape, (2, 53))
        self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))
        expected = encoder.predictor(encoder.transform.history(levels))
        torch.testing.assert_close(features[:, :53], expected)
        validate_config(self.config, encoder.transform)
        bad = dict(self.config, evaluation_seeds=self.config['validation_seeds'])
        with self.assertRaises(ValueError):
            validate_config(bad, encoder.transform)

    def test_actor_gradient_passes_through_frozen_critic(self):
        torch.manual_seed(1)
        actor = Actor()
        critic = Critic().requires_grad_(False)
        before = state_hash(critic)
        actor_before = state_hash(actor)
        features = torch.randn(4, FEATURES)
        last = torch.full((4, 53), 50.0)
        nominal = last[:, 41:]
        config = dict(self.config, action_weight=0.0, regularization_weight=0.0, residual_weight=0.0)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        metrics = actor_update(actor, critic, optimizer, features, last, last[:, 41:52], config, nominal)
        self.assertNotEqual(state_hash(actor), actor_before)
        self.assertEqual(state_hash(critic), before)
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))
        self.assertAlmostEqual(metrics['actor_loss'], metrics['actor_q_loss'] + metrics['actor_movement'] + metrics['actor_window'])
        action = actor(features, nominal, config)
        self.assertTrue(((action - nominal[:, :11]).abs() <= config['residual_limit']).all())

    def test_streak_breaks_on_any_out_of_band_step(self):
        config = dict(self.config, target_streak=3)
        result = streak_metrics([161, 161, 170, 159, 160, 161, 165], config)
        self.assertEqual(result['longest_streak'], 4)
        self.assertEqual(result['streak_start'], 4)
        self.assertTrue(result['streak_success'])
        self.assertAlmostEqual(result['streak_rmse'], np.sqrt(2 / 3))
        self.assertFalse(streak_metrics([160, 170, 160], config)['streak_success'])

    def test_conditioned_forecast_has_a_gradient_from_command(self):
        source = torch.load(ROOT / self.config['checkpoint'], weights_only=False, map_location='cpu')
        model = ControlledForecast(source, conditioned=True)
        with torch.no_grad():
            model.command_head[0].weight.zero_()
            model.command_head[0].bias.zero_()
            model.command_head[0].weight[0, -11] = 1
            model.command_head[-1].weight[0, 0] = 1
        levels = torch.full((1, model.transform.window + 1, 53), 50.0)
        command = torch.full((1, 12), 50.0, requires_grad=True)
        predictions = model.forecast(levels, command)
        self.assertEqual(predictions.shape, (1, 3, 53))
        gradient, = torch.autograd.grad(predictions[0, 0, 0], command)
        self.assertGreater(gradient[0, 0].item(), 0)

    def test_movement_penalty_is_separate_and_positive(self):
        actor = Actor()
        with torch.no_grad():
            actor.head[-1].bias.fill_(0.2)
        critic = Critic().requires_grad_(False)
        last = torch.full((4, 53), 50.0)
        metrics = actor_update(actor, critic, torch.optim.SGD(actor.parameters(), lr=0.001),
                               torch.zeros(4, FEATURES), last, last[:, 41:52], self.config, last[:, 41:])
        self.assertGreater(metrics['actor_movement'], 0)
        self.assertAlmostEqual(metrics['actor_movement'], self.config['action_weight'] * metrics['actor_delta'], places=6)

    def test_absorbing_failure_takes_effect_after_three_identical_steps(self):
        config = dict(self.config, tracking_weight=0.0, barrier_weight=0.0, gamma=0.5, failure_penalty=10.0)
        future = torch.zeros(1, 2, 5, 53)
        future[0, 1, :, 0] = torch.arange(1., 6.)
        data = dict(future=future, levels=torch.zeros(1, 61, 53),
                    lower=torch.zeros(1, 52), upper=torch.full((1, 52), 100.0))
        self.assertAlmostEqual(returns(data, config)[0, 0].item(), -10 * (0.25 + 0.125 + 0.0625) / 1.9375, places=6)
        self.assertEqual(returns(data, config)[0, 1].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
