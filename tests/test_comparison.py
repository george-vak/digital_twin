import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline import PersistenceModel
from compare import average_change_summary, baseline_mae, comparison_table, relative_path
from eval import data_fingerprint, report_metadata_path
from model import ROOT, GRUModel, WindowDataset


class ComparisonTests(unittest.TestCase):
    def test_project_relative_paths(self):
        self.assertEqual(relative_path(ROOT / "models/gru.pt"), "models/gru.pt")
        self.assertEqual(relative_path(ROOT / "data/base_20_5"), "data/base_20_5")

    def test_last_observation_and_horizon(self):
        frame = pd.DataFrame({
            "run_id": ["a"] * 8 + ["b"] * 8,
            "step": list(range(8)) * 2,
            "elapsed_seconds": list(np.arange(8) * 60) * 2,
            "CV[0]": list(range(8)) + list(range(100, 108)),
        })
        for horizon in (1, 3):
            dataset = WindowDataset(frame, ["CV[0]"], 3, horizon, 60)
            x, y = dataset[0]
            self.assertEqual(PersistenceModel()(x.unsqueeze(0)).item(), 2)
            self.assertEqual(y.item(), 2 + horizon)
            self.assertEqual(len(dataset), 2 * (8 - 3 - horizon + 1))
            np.testing.assert_allclose(baseline_mae(dataset, np.array([2.])), [2 * horizon])

    def test_improvement_sign_and_zero_denominator(self):
        table = comparison_table(pd.DataFrame({"MAE": [7., 12., 0., 2.]}), np.array([10., 10., 0., 0.]))
        np.testing.assert_allclose(table["Улучшение MAE"], [3, -2, 0, -2])
        np.testing.assert_allclose(table["Улучшение, %"][:2], [30, -20])
        self.assertTrue(table["Улучшение, %"][2:].isna().all())

    def test_fingerprint_includes_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "test.csv"
            file.write_text("CV[0]\n1\n")
            (root / "metadata.json").write_text(json.dumps({"preprocessing": {"parameters_file": "normalization.json"}}))
            scaler = root / "normalization.json"
            scaler.write_text('{"scale": 1}')
            before = data_fingerprint(file)
            scaler.write_text('{"scale": 2}')
            self.assertNotEqual(before, data_fingerprint(file))

    def test_group_averages_exclude_ties_and_undefined_percentages(self):
        table = comparison_table(
            pd.DataFrame({"MAE": [7., 9., 12., 16., 10., 0., 2.]}),
            np.array([10., 10., 10., 10., 10., 0., 0.]),
        )
        summary = average_change_summary(table)
        self.assertIn("снижение MAE на 20.00% (2 переменных)", summary)
        self.assertIn("рост MAE на 40.00% (2 переменных)", summary)
        self.assertIn("без процентной оценки: 2", summary)
        empty_groups = average_change_summary(table.iloc[4:])
        self.assertIn("снижение MAE: нет переменных", empty_groups)
        self.assertIn("рост MAE: нет переменных", empty_groups)

    def test_cli_report_and_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({
                "run_id": ["a"] * 8 + ["b"] * 8,
                "step": list(range(8)) * 2,
                "elapsed_seconds": list(np.arange(8) * 60) * 2,
                "CV[0]": list(range(8)) + list(range(100, 108)),
                "MV[0]": [5.] * 16,
            })
            data = root / "test.csv"
            frame.to_csv(data, index=False)
            model = GRUModel(2, 2)
            for parameter in model.parameters():
                parameter.data.zero_()
            checkpoint = root / "toy.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "config": {"input_size": 2, "hidden_size": 2, "window": 3, "horizon": 3, "frequency_seconds": 60},
                "normalization": {"columns": ["CV[0]", "MV[0]"], "mean": {"CV[0]": 10., "MV[0]": 5.}, "scale": {"CV[0]": 2., "MV[0]": 1.}},
                "variables": {c: {"description": c, "unit": "unit"} for c in ("CV[0]", "MV[0]")},
                "training": {"rows": {"train": 16, "validation": 8}, "dataset": "synthetic", "best_epoch": 1},
            }, checkpoint)
            report = root / "gru.csv"
            output = root / "comp/comparison.csv"
            subprocess.run([sys.executable, str(ROOT / "eval.py"), str(data), "--model", str(checkpoint), "--output", str(report)], check=True, capture_output=True)
            self.assertTrue(report_metadata_path(report).is_file())
            with report.open(encoding="utf-8-sig", newline="") as file:
                report_rows = list(csv.reader(file))
            report_header = next(i for i, row in enumerate(report_rows) if row and row[0] == "Параметр")
            report_preamble = [row for row in report_rows[:report_header] if row]
            self.assertTrue(all(len(row) == 1 and ": " in row[0] for row in report_preamble))
            self.assertTrue(all("SHA256" not in row[0] for row in report_preamble))
            self.assertIn([f"Модель: {relative_path(checkpoint)}"], report_preamble)
            self.assertTrue(all(len(row) == 4 for row in report_rows[report_header:]))
            command = [sys.executable, str(ROOT / "compare.py"), str(report), "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True)
            with output.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file))
            header = next(i for i, row in enumerate(rows) if row and row[0] == "Параметр")
            preamble = [row for row in rows[:header] if row]
            self.assertTrue(all(len(row) == 1 and ": " in row[0] for row in preamble))
            self.assertTrue(all("SHA256" not in row[0] for row in preamble))
            self.assertTrue(all(not row[0].startswith(("Прогноз:", "Baseline:", "Метрика:", "Нормализация:", "Сопоставимость:", "Границы окон:", "Абсолютное улучшение:", "Улучшение, %:", "Нулевая ошибка baseline:")) for row in preamble))
            self.assertIn([f"Файл оценки: {relative_path(data)}"], preamble)
            self.assertTrue(preamble[-1][0].startswith("Среднее изменение по группам:"))
            self.assertTrue(all(len(row) == 7 for row in rows[header:]))
            metrics = {row[0]: row for row in rows[header + 1:] if row[0] in ("CV[0]", "MV[0]")}
            np.testing.assert_allclose(list(map(float, metrics["CV[0]"][3:6])), [50, 3, -47])
            self.assertEqual(metrics["MV[0]"][-1], "—")
            original_output = output.read_bytes()
            original_report = report.read_bytes()
            report.write_bytes(original_report + b"\n")
            changed_report = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(changed_report.returncode, 0)
            self.assertIn("GRU report changed", changed_report.stderr)
            report.write_bytes(original_report)
            frame.loc[0, "CV[0]"] += 1
            frame.to_csv(data, index=False)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Data or its metadata/normalization changed", result.stderr)
            self.assertEqual(output.read_bytes(), original_output)


if __name__ == "__main__":
    unittest.main()
