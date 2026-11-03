import unittest
import numpy as np
import torch
from exp_6.safety import fit_bounds,ReleaseGate,barrier
from exp_6.data import Environment,Replay
from exp_6.networks import Actor,Dynamics,Transform,adapt_state,actor_update,tracking_loss
from exp_3.networks import Actor as OldActor,Dynamics as OldDynamics
from model import GRUModel


class Experiment6Tests(unittest.TestCase):
    def source(self):
        cols=[f'CV[{i}]' for i in range(41)]+[f'MV[{i}]' for i in range(12)]
        return dict(config=dict(window=3,frequency_seconds=60,hidden_size=4),
                    state_dict=GRUModel(53,4).state_dict(),
                    normalization=dict(mean=dict.fromkeys(cols,0.),scale=dict.fromkeys(cols,1.)))

    def test_tracking_is_per_transition_and_has_correct_gradient(self):
        costs=torch.tensor([150.,160.,170.],dtype=torch.float64,requires_grad=True)
        loss=tracking_loss(costs,160.)
        self.assertAlmostEqual(float(loss.detach()),200/3)
        loss.backward()
        self.assertLess(costs.grad[0],0)
        self.assertEqual(costs.grad[1],0)
        self.assertGreater(costs.grad[2],0)
        self.assertTrue(torch.autograd.gradcheck(lambda x:tracking_loss(x,160.),(costs,)))
        self.assertGreater(tracking_loss(torch.tensor([150.,170.]),160.),0)

    def test_actor_update_moves_cost_towards_target_on_both_sides(self):
        from types import SimpleNamespace
        class Policy(torch.nn.Module):
            def __init__(self,value):
                super().__init__();self.value=torch.nn.Parameter(torch.tensor(value))
            def forward(self,history,last,transform):
                return self.value.expand(1,11)
        class Model(torch.nn.Module):
            def forward(self,history,last,action,transform):
                basis=action.new_zeros(41);basis[19]=1/.0536
                return action[:,:1]*basis
        transform=SimpleNamespace(frequency=60,cv_levels=lambda last,delta:delta)
        batch=(torch.zeros(1,3,52),torch.zeros(1,53),None,None,torch.zeros(1,11),
               torch.full((1,52),-10000.),torch.full((1,52),10000.))
        config=dict(target_cost=160.,regularization_scale=100.,regularization_weight=0.,
                    barrier_weight=0.,barrier_epsilon=.1)
        for initial in (150.,170.):
            actor=Policy(initial);model=Model()
            metrics=actor_update(actor,model,transform,torch.optim.SGD(actor.parameters(),lr=.1),batch,config)
            self.assertLess(abs(float(actor.value.detach())-160),abs(initial-160))
            self.assertAlmostEqual(metrics['actor_tracking_loss'],100.,places=2)
            self.assertAlmostEqual(metrics['actor_loss'],metrics['actor_tracking_loss'])

    def test_raw_quantiles_and_no_median_symmetry(self):
        x=np.zeros((1000,53))
        x[:,:52]=((np.arange(1000)/100)**2)[:,None];x[:,52]=50
        b=fit_bounds(x)
        np.testing.assert_allclose(b['lower'],np.quantile(x[:,:52],.01,axis=0))
        np.testing.assert_allclose(b['upper'],np.quantile(x[:,:52],.99,axis=0))
        self.assertNotAlmostEqual((b['lower'][0]+b['upper'][0])/2,np.median(x[:,0]))
        self.assertEqual(len(b['lower']),52)

    def test_gate_excludes_mv11_and_returns_immediately(self):
        b=dict(lower=[1.]*52,upper=[3.]*52)
        gate=ReleaseGate(b);current=np.full(53,2.);current[52]=999
        self.assertFalse(gate.decide(current,np.full(11,2.))[0])
        current[0]=.5
        self.assertTrue(gate.decide(current,np.full(11,2.))[0])
        current[0]=2
        self.assertFalse(gate.decide(current,np.full(11,2.))[0])
        candidate=np.full(11,2.);candidate[10]=4
        self.assertIn('candidate_MV[10]',gate.decide(current,candidate)[1])

    def test_barrier_gradient_inside_outside_and_point(self):
        x=torch.tensor([-.5,.3,1.5],dtype=torch.float64,requires_grad=True)
        low=torch.zeros_like(x);high=torch.ones_like(x)
        self.assertTrue(torch.autograd.gradcheck(lambda v:barrier(v,low,high),(x,)))
        loss=barrier(x,low,high);loss.backward()
        self.assertLess(x.grad[0],0);self.assertGreater(x.grad[2],0)
        self.assertTrue(torch.isfinite(barrier(x,low,low)))

    def test_full_warmup_not_only_history(self):
        class Backend:
            def reset(self,**kwargs):
                self.window=kwargs['window']
                values=np.tile(np.arange(1000)[:,None],(1,53)).astype(float);values[:,52]=50
                return values
        backend=Backend();env=Environment(backend,dict(warmup_steps=1000,frequency_seconds=60),60)
        history=env.reset(1)
        self.assertEqual(backend.window,999);self.assertEqual(history.shape,(61,53))
        self.assertAlmostEqual(env.bounds['lower'][0],9.99)
        self.assertAlmostEqual(env.bounds['upper'][0],989.01)

    def test_weight_transfer_removal_replay_and_actor_update(self):
        source=self.source();t=Transform(source);actor=Actor(source);model=Dynamics(source)
        self.assertEqual(t.mean.shape,(52,));self.assertEqual(t.scale.shape,(52,))
        old_actor=OldActor(source);old_model=OldDynamics(source)
        actor.load_state_dict(adapt_state(old_actor.state_dict(),True,4))
        model.load_state_dict(adapt_state(old_model.state_dict(),False,4))
        self.assertEqual(actor.gru.input_size,52);self.assertEqual(actor.head.out_features,11)
        self.assertEqual(model.head.in_features,15)
        levels=np.full((4,53),2.,dtype=np.float32)
        other=levels.copy();other[:,52]=np.arange(4)*100
        torch.testing.assert_close(t.history(torch.tensor(levels)),t.history(torch.tensor(other)))
        replay=Replay(10,42,3)
        for value in (1.,3.):
            bounds=dict(lower=[value]*52,upper=[value+2]*52)
            replay.add(levels,np.full(11,value),levels[-1],bounds)
        batch=replay.sample(2,t)
        torch.testing.assert_close(batch[2][:,0],batch[5][:,0])
        self.assertEqual(batch[0].shape,(2,3,52));self.assertEqual(batch[4].shape,(2,11))
        before=[p.detach().clone() for p in actor.parameters()]
        metrics=actor_update(actor,model,t,torch.optim.Adam(actor.parameters(),lr=1e-4),batch,
                            dict(target_cost=160.,regularization_scale=100,regularization_weight=10000,barrier_epsilon=.1,barrier_weight=1))
        self.assertTrue(np.isfinite(metrics['actor_loss']))
        self.assertTrue(any(not torch.equal(p,v) for p,v in zip(actor.parameters(),before)))


if __name__=='__main__':unittest.main()
