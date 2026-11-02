import unittest
import ast
from pathlib import Path
import pandas as pd
from exp_3.networks import cost
import numpy as np
import torch
from exp_5.safety import barrier, fit_bounds, ReleaseGate


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.bounds=dict(center=[0.]*53,p80_radius=[1.]*53,p90_radius=[2.]*53)

    def test_cost_matches_original_and_has_correct_derivative(self):
        module=ast.parse((Path(__file__).resolve().parents[1]/'temain.py').read_text())
        cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='TemainAnalyser')
        namespace={'pd':pd,'np':np}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'temain.py','exec'),namespace)
        x=torch.linspace(1,30,82,dtype=torch.float64).reshape(2,41).requires_grad_()
        for frequency in (30,60,120):
            expected=namespace['TemainAnalyser'](freq=f'{frequency}s')(x.detach().numpy())
            np.testing.assert_allclose(cost(x,frequency).detach().numpy(),expected,rtol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(cost,(x,)))

    def test_release_on_cv_and_recovery_hysteresis(self):
        gate=ReleaseGate(self.bounds,min_release_steps=2,recovery_steps=2)
        current=np.zeros(53); current[0]=-3
        self.assertTrue(gate.decide(current,np.zeros(12))[0])
        current[0]=1.5
        self.assertTrue(gate.decide(current,np.zeros(12))[0])
        current[0]=0
        self.assertTrue(gate.decide(current,np.zeros(12))[0])
        self.assertFalse(gate.decide(current,np.zeros(12))[0])

    def test_two_separate_percentile_gates(self):
        current=np.zeros(53); current[0]=1.5
        action=np.zeros(12)
        self.assertTrue(ReleaseGate(self.bounds,percentile=80).decide(current,action)[0])
        self.assertFalse(ReleaseGate(self.bounds,percentile=90).decide(current,action)[0])
        gate=ReleaseGate(self.bounds,min_release_steps=1,recovery_steps=1,percentile=90)
        current[0]=3
        self.assertTrue(gate.decide(current,action)[0])
        current[0]=1.5
        self.assertFalse(gate.decide(current,action)[0])

    def test_candidate_checked_before_application(self):
        gate=ReleaseGate(self.bounds)
        action=np.zeros(12); action[3]=3
        released,reason=gate.decide(np.zeros(53),action)
        self.assertTrue(released)
        self.assertIn('candidate_MV[3]',reason)

    def test_barrier_gradient_inside_and_outside(self):
        x=torch.tensor([-.2,.3,1.5],dtype=torch.float64,requires_grad=True)
        center=torch.zeros_like(x); radius=torch.ones_like(x)
        self.assertTrue(torch.autograd.gradcheck(lambda v:barrier(v,center,radius),(x,)))
        barrier(x,center,radius).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertLess(x.grad[0],0)
        self.assertTrue((x.grad[1:]>0).all())

    def test_constant_channels_have_finite_nonzero_bounds(self):
        b=fit_bounds(np.full((100,53),50.))
        self.assertTrue(np.all(np.array(b['p80_radius'])>0))
        self.assertTrue(np.all(np.array(b['p90_radius'])>b['p80_radius']))


if __name__=='__main__': unittest.main()
