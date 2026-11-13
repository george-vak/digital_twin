import unittest
import numpy as np
from exp_7.data import Environment
from exp_7.safety import fit_bounds, cv_violations


class Experiment7Tests(unittest.TestCase):
    def test_cv_only_bounds(self):
        warmup=np.tile(np.arange(1000)[:,None],(1,53)).astype(float)
        warmup[:,52]=50
        b=fit_bounds(warmup)
        np.testing.assert_allclose(b['lower'][:41],9.99)
        np.testing.assert_allclose(b['upper'][:41],989.01)
        np.testing.assert_allclose(b['lower'][41:],0)
        np.testing.assert_allclose(b['upper'][41:],100)
        row=np.full(53,500.)
        self.assertEqual(cv_violations(row,b),'')
        row[0]=0;row[40]=1000
        self.assertEqual(cv_violations(row,b),'CV[0];CV[40]')

    def test_terminal_transition_keeps_actor_action(self):
        class Backend:
            def step(self,action):
                self.action=action.copy()
                row=np.full(53,.5);row[4]=1.1
                return row,160.
        backend=Backend()
        env=Environment(backend,dict(freeze_atol=1e-10,freeze_steps=3,episode_steps=1),2)
        env.levels=np.full((3,53),.5);env.frozen=0;env.steps=0
        env.bounds=dict(lower=[0.]*52,upper=[1.]*52)
        action=np.full(12,50.)
        following,cost,terminated,truncated=env.step(action)
        self.assertTrue(terminated);self.assertFalse(truncated)
        self.assertEqual(env.termination_reason,'CV[4]')
        self.assertEqual(following[-1,4],1.1)
        np.testing.assert_array_equal(backend.action,action)
        with self.assertRaises(ValueError):env.step(None)



class SurvivalObjectiveTests(unittest.TestCase):
    def test_initial_violation_ends_episode_without_action(self):
        class Backend:
            def reset(self, **kwargs):
                rows = np.tile(np.arange(1000)[:, None], (1, 53)).astype(float)
                rows[:, 52] = 50
                return rows
            def step(self, action):
                raise AssertionError('Invalid initial state must not receive an action')
        env = Environment(Backend(), dict(warmup_steps=1000, frequency_seconds=60), 60)
        env.reset(1)
        self.assertTrue(env.done)
        self.assertEqual(env.steps, 0)
        self.assertTrue(env.cv_violation)
        with self.assertRaises(ValueError):
            env.step(np.zeros(12))

    def test_safe_episode_reaches_exactly_200_steps(self):
        class Backend:
            def step(self, action):
                return np.full(53, .5), 160.
        env = Environment(Backend(), dict(freeze_atol=1e-10, freeze_steps=201, episode_steps=200), 2)
        env.levels = np.full((3, 53), .5)
        env.steps = env.frozen = 0
        env.bounds = dict(lower=[0.]*52, upper=[1.]*52)
        for step in range(200):
            _, _, terminated, truncated = env.step(np.zeros(12))
            self.assertFalse(terminated)
            self.assertEqual(truncated, step == 199)
        with self.assertRaises(ValueError):
            env.step(np.zeros(12))

    def setup_rollout(self, value):
        import torch
        from types import SimpleNamespace
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(value))
            def forward(self, history, last, transform):
                return self.value.expand(len(last), 11)
        class Model(torch.nn.Module):
            def forward(self, history, last, action, transform):
                basis = action.new_zeros(41)
                basis[19] = 1/.0536
                return action[:, :1]*basis
        transform = SimpleNamespace(frequency=60, cv_levels=lambda last, delta:delta,
                                    history=lambda levels:levels[:, 1:, :52]-levels[:, :-1, :52])
        config = dict(target_cost=160., episode_steps=200, cost_scale=10., survival_temperature=.05,
                      survival_cutoff=1e-6, survival_weight=1., tracking_weight=1.,
                      regularization_scale=100., regularization_weight=0., barrier_weight=0., barrier_epsilon=.1)
        levels = torch.zeros(1, 4, 53)
        lower, upper = torch.full((1, 52), -10000.), torch.full((1, 52), 10000.)
        return Policy(), Model(), transform, config, levels, lower, upper

    def test_horizon_and_target_both_affect_return(self):
        import torch
        from exp_7.networks import rollout_returns
        actor, model, transform, config, levels, lower, upper = self.setup_rollout(160.)
        long_steps, long_tracking = rollout_returns(actor, model, transform, levels, lower, upper, torch.tensor([200]), config)
        short_steps, short_tracking = rollout_returns(actor, model, transform, levels, lower, upper, torch.tensor([10]), config)
        self.assertAlmostEqual(long_steps.item(), 200., delta=.05)
        self.assertAlmostEqual(long_tracking.item(), 200., delta=.05)
        self.assertGreater(long_steps.item(), short_steps.item())
        with torch.no_grad():
            actor.value.fill_(170.)
        _, off_target = rollout_returns(actor, model, transform, levels, lower, upper, torch.tensor([10]), config)
        self.assertLess(off_target.item(), short_tracking.item())
        upper[:, 19] = 1.
        failed_steps, _ = rollout_returns(actor, model, transform, levels, lower, upper, torch.tensor([200]), config)
        self.assertLess(failed_steps.item(), 1.)

    def test_update_moves_towards_160_and_logs_complete_objective(self):
        import torch
        from exp_7.networks import actor_update
        for value in (150., 170.):
            actor, model, transform, config, levels, lower, upper = self.setup_rollout(value)
            batch = (transform.history(levels), levels[:, -1], None, None,
                     torch.zeros(1, 11), lower, upper, levels, torch.tensor([3]))
            metrics = actor_update(actor, model, transform, torch.optim.SGD(actor.parameters(),lr=.1),batch,config)
            self.assertLess(abs(actor.value.item()-160), abs(value-160))
            self.assertAlmostEqual(metrics['actor_loss'], metrics['actor_survival_loss']+
                                   metrics['actor_tracking_objective']+metrics['actor_regularization']+
                                   metrics['actor_barrier'], places=5)
            self.assertGreater(metrics['actor_expected_safe_steps'], 2.99)

    def test_survival_gradient_pushes_inside_barrier(self):
        import torch
        from exp_7.networks import survival_probability
        cv = torch.tensor([[1.02, .5]], requires_grad=True)
        probability = survival_probability(cv, torch.zeros_like(cv), torch.ones_like(cv), .05)
        probability.backward()
        self.assertLess(cv.grad[0, 0].item(), 0)
        self.assertTrue(torch.isfinite(cv.grad).all())


if __name__ == '__main__':
    unittest.main()
