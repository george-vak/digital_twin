import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiment_paths import DEFAULT_DATA, DEFAULT_RUN, ROOT, resolve_path, run_for_artifact, training_data_path


class ExperimentLayoutTests(unittest.TestCase):
    def test_legacy_training_label_and_run_isolation(self):
        self.assertEqual(training_data_path("/old/checkout/data/base_20_5", DEFAULT_RUN / "models/gru.pt"), DEFAULT_DATA)
        self.assertEqual(run_for_artifact(ROOT / "exp_2/runs/seed_42/models/gru.pt"), ROOT / "exp_2/runs/seed_42")
        self.assertEqual(resolve_path("exp_1/data/base_20_5"), DEFAULT_DATA)

    def test_dry_run_from_another_working_directory(self):
        target = ROOT / "exp_1/runs/layout_test_dry_run"
        self.assertFalse(target.exists())
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / "exp_1/run.py"), "--run", target.name, "--dry-run"],
                                    cwd=tmp, capture_output=True, text=True, check=True)
        self.assertFalse(target.exists())
        self.assertIn(str(target / "models/gru.pt"), result.stdout)
        self.assertIn(str(DEFAULT_DATA / "test.csv.gz"), result.stdout)
        self.assertNotIn("/model.py", result.stdout)  # default is evaluation, not training

    def test_existing_run_cannot_be_overwritten(self):
        result = subprocess.run([sys.executable, str(ROOT / "exp_1/run.py"), "--run", "original", "--train"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Run already exists", result.stderr)

    def test_archived_reports_are_portable_and_verified(self):
        from compare import read_report
        from eval import data_fingerprint, file_sha256
        for report in (DEFAULT_RUN / "results").glob("*.csv"):
            with self.subTest(report=report.name):
                details, table = read_report(report)
                self.assertEqual(len(table), 53)
                for key in ("Модель", "Файл оценки", "Датасет обучения"):
                    self.assertFalse(Path(details[key]).is_absolute())
                    self.assertTrue(resolve_path(details[key]).exists())
                self.assertEqual(file_sha256(resolve_path(details["Модель"])), details["SHA256 модели"])
                self.assertEqual(data_fingerprint(resolve_path(details["Файл оценки"])), details["SHA256 данных"])


if __name__ == "__main__":
    unittest.main()
