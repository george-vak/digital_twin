import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp_7.data import Replay
from exp_7.run import replay_signature, seed_replay
from exp_7.tune import rank, validation_metrics


class HyperparameterSearchTests(unittest.TestCase):
    def test_validation_prioritizes_joint_target_then_safe_duration(self):
        target = dict(name='target',mean_safe_target_steps=1.,mean_safe_steps=3.,
                      safe_price_score=1.,cost_mae=10.)
        longer = dict(name='longer',mean_safe_target_steps=0.,mean_safe_steps=100.,
                      safe_price_score=80.,cost_mae=0.)
        self.assertLess(rank(target),rank(longer))
        short = dict(target,name='short',mean_safe_steps=2.,cost_mae=0.)
        self.assertLess(rank(target),rank(short))

    def test_scoring_excludes_unsafe_steps_and_includes_zero_length_episodes(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            summary = dict(episodes=[{},{}],safe_target_steps=1,mean_safe_steps=.5,
                           mean_steps=1.,completed_episodes=0,cost_mae=0.,initial_cv_outside=1)
            (directory/'summary.json').write_text(json.dumps(summary))
            pd.DataFrame(dict(cost=[160.,160.],safe_step=[True,False])).to_csv(directory/'steps.csv',index=False)
            metrics = validation_metrics(directory)
            self.assertEqual(metrics['mean_safe_target_steps'],.5)
            self.assertEqual(metrics['safe_price_score'],.5)

    def test_cache_roundtrip_and_rejection_after_source_change(self):
        with tempfile.TemporaryDirectory() as name, patch('exp_7.run.ROOT',Path(name)):
            directory = Path(name)
            for filename in ('replay.csv','parent.pt','source.pt'):
                (directory/filename).write_text('fixture')
            config = dict(replay_source='replay.csv',parent_controller='parent.pt',checkpoint='source.pt',
                          replay_cache='replay.npz',warmup_steps=1000,frequency_seconds=60,episode_steps=200)
            levels = np.full((1,4,53),.5,dtype=np.float32)
            np.savez(directory/'replay.npz',levels=levels,action=np.zeros((1,11)),
                     following=levels[:,-1],lower=np.zeros((1,52)),upper=np.ones((1,52)),
                     remaining=np.array([200]),signature=json.dumps(replay_signature(config)))
            replay = Replay(10,42,3)
            seed_replay(None,replay,config,None,None)
            self.assertEqual(len(replay.rows),1)
            np.testing.assert_array_equal(replay.rows[0][0],levels[0])
            (directory/'parent.pt').write_text('changed model')
            with self.assertRaisesRegex(ValueError,'does not match'):
                seed_replay(None,replay,config,None,None)


if __name__ == '__main__':
    unittest.main()
