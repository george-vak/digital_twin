import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from evaluate_mase import mase_scale, predict_levels, scaled_error_statistics
from generate import fit_normalization
from model import GRUModel, load_physical_frame
from preprocess import difference_frame, prepare


class MaseTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "run_id": ["a"] * 8 + ["b"] * 8,
            "step": list(range(8)) * 2, "elapsed_seconds": list(np.arange(8) * 60) * 2,
            "CV[0]": list(np.arange(8) * 2.) + list(100 + np.arange(8) * 4.),
            "MV[0]": [5.] * 16,
        })

    def test_differences_skip_run_boundaries_and_keep_target_keys(self):
        delta = difference_frame(self.frame(), ["CV[0]", "MV[0]"])
        np.testing.assert_array_equal(delta["CV[0]"], [2.] * 7 + [4.] * 7)
        self.assertEqual(delta["step"].tolist(), list(range(1, 8)) * 2)
        np.testing.assert_array_equal(mase_scale(self.frame(), ["CV[0]", "MV[0]"], [2., 1.]), [1.5, 0.])
        broken = self.frame().drop(index=2)
        with self.assertRaises(ValueError):
            difference_frame(broken, ["CV[0]", "MV[0]"])

    def test_mase_std_and_scale_invariance_with_constant_series(self):
        errors = np.array([[0., 1.], [-3., 2.], [6., 3.]])
        mean, std, mae = scaled_error_statistics(errors, np.array([3., 0.]))
        self.assertEqual(mean[0], 1.)
        self.assertAlmostEqual(std[0], np.sqrt(2 / 3))
        self.assertTrue(np.isnan(mean[1]) and np.isnan(std[1]))
        self.assertEqual(mae[1], 2.)
        other = scaled_error_statistics(errors / [5., 10.], np.array([3., 0.]) / [5., 10.])
        np.testing.assert_allclose(other[0], mean, equal_nan=True)
        np.testing.assert_allclose(other[1], std, equal_nan=True)

    def test_inverse_difference_uses_last_observation_and_common_targets(self):
        columns = ["CV[0]", "MV[0]"]
        frame = self.frame()
        network = GRUModel(2, 2)
        for parameter in network.parameters():
            parameter.data.zero_()
        checkpoint = {
            "config": {"input_size": 2, "hidden_size": 2, "window": 2, "horizon": 1,
                       "frequency_seconds": 60, "target_transform": "first_difference"},
            "normalization": {"columns": columns, "mean": {"CV[0]": 3., "MV[0]": 0.},
                              "scale": {"CV[0]": 1., "MV[0]": 1.}},
            "state_dict": network.state_dict(),
        }
        prediction = predict_levels(checkpoint, frame)
        self.assertEqual(len(prediction), 10)
        self.assertEqual(prediction.index.tolist()[:5], [("a", i) for i in range(3, 8)])
        self.assertEqual(prediction.loc[("a", 3), "CV[0]"], 7.)  # observed 4 + predicted delta 3
        self.assertEqual(prediction.loc[("b", 3), "CV[0]"], 111.)  # observed 108 + delta 3
        np.testing.assert_array_equal(prediction["MV[0]"], 5.)
        changed = frame.copy()
        changed.loc[3, "CV[0]"] = 999.
        self.assertEqual(predict_levels(checkpoint, changed).loc[("a", 3), "CV[0]"], 7.)
        checkpoint["config"]["target_transform"] = "levels"
        levels = predict_levels(checkpoint, frame)
        self.assertEqual(len(levels), 12)
        self.assertEqual(len(levels.index.intersection(prediction.index)), 10)

    def test_preprocessing_fits_only_train_and_can_be_inverted(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "prepared"
            source.mkdir()
            frame, columns = self.frame(), ["CV[0]", "MV[0]"]
            raw_norm = fit_normalization(frame, columns)
            (source / "normalization.json").write_text(json.dumps(raw_norm))
            (source / "metadata.json").write_text(json.dumps({
                "frequency_seconds": 60,
                "splits": {split: {"rows": 16, "steps_per_run": 8} for split in ("train", "validation", "test")},
            }))
            pd.DataFrame({"variable": columns, "description": columns, "unit": ["unit"] * 2}).to_csv(source / "variables.csv", index=False)
            for split, multiplier in (("train", 1), ("validation", 10), ("test", 20)):
                values = frame.copy()
                values["CV[0]"] *= multiplier
                values.to_csv(source / f"{split}.csv.gz", index=False)
            prepare(source, output)
            norm = json.loads((output / "normalization.json").read_text())
            self.assertEqual(norm["mean"]["CV[0]"], 3.)
            self.assertEqual(norm["scale"]["CV[0]"], 1.)
            standardized = pd.read_csv(output / "train.csv.gz")
            self.assertAlmostEqual(standardized["CV[0]"].mean(), 0.)
            self.assertAlmostEqual(standardized["CV[0]"].std(ddof=0), 1.)
            restored = load_physical_frame(output / "test.csv.gz", columns, 60)
            np.testing.assert_allclose(restored["CV[0]"], [40.] * 7 + [80.] * 7)


if __name__ == "__main__":
    unittest.main()
