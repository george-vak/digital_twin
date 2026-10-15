"""Compare a saved GRU evaluation report with persistence on the same data."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from baseline import PersistenceModel
from eval import data_fingerprint, display_details, file_sha256, relative_path, report_metadata_path
from model import ROOT, WindowDataset, load_frame


def read_report(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    header = ["Параметр", "Описание", "Единица", "MAE"]
    start = rows.index(header)
    metadata = report_metadata_path(path)
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["report_sha256"] != file_sha256(path):
            raise ValueError("GRU report changed. Run eval.py again.")
        details = saved["details"]
    else:
        # Legacy reports kept their verification fields in the CSV itself.
        details = {row[0]: row[1] for row in rows[:start] if len(row) > 1 and row[0]}
    table = pd.DataFrame(rows[start + 1:], columns=header)
    table["MAE"] = pd.to_numeric(table["MAE"])
    return details, table


@torch.inference_mode()
def baseline_mae(dataset, scale):
    model = PersistenceModel()
    errors = np.zeros(len(scale), dtype=np.float64)
    for x, y in DataLoader(dataset, batch_size=256):
        errors += (np.abs(model(x).numpy().astype(np.float64) - y.numpy()) * scale).sum(axis=0)
    return errors / len(dataset)


def comparison_table(gru_table, baseline_errors):
    table = gru_table.rename(columns={"MAE": "MAE GRU"}).copy()
    table["MAE baseline"] = baseline_errors
    table["Улучшение MAE"] = table["MAE baseline"] - table["MAE GRU"]
    table["Улучшение, %"] = 100 * table["Улучшение MAE"] / table["MAE baseline"].replace(0, np.nan)
    return table


def average_change_summary(table):
    """Average relative changes within each group; each variable has equal weight."""
    percent = table["Улучшение, %"]
    better = percent[percent > 0]
    worse = -percent[percent < 0]
    parts = []
    for label, values in (("снижение MAE", better), ("рост MAE", worse)):
        if len(values):
            parts.append(f"{label} на {values.mean():.2f}% ({len(values)} переменных)")
        else:
            parts.append(f"{label}: нет переменных")
    parts.append(f"без процентной оценки: {percent.isna().sum()} (MAE baseline = 0)")
    return "; ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="CSV report produced by eval.py")
    parser.add_argument("--output", type=Path, help="Override comp/comparison_<report>.csv")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "comp" / f"comparison_{args.report.stem}.csv"
    if args.output.resolve() == args.report.resolve():
        parser.error("Comparison output must differ from the GRU report")

    try:
        details, gru_table = read_report(args.report)
    except ValueError as error:
        parser.error(str(error))
    if not {"SHA256 данных", "SHA256 модели", "Размер окна, точек"} <= details.keys():
        parser.error("Missing verification metadata. Restore the matching .metadata file or run eval.py again.")
    data_path = Path(details["Файл оценки"])
    model_path = Path(details["Модель"])
    if data_fingerprint(data_path) != details["SHA256 данных"]:
        parser.error("Data or its metadata/normalization changed. Run eval.py again.")
    if file_sha256(model_path) != details["SHA256 модели"]:
        parser.error("Model checkpoint changed. Run eval.py again.")

    torch.set_num_threads(4)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    expected = (config["window"], config["horizon"], config["frequency_seconds"])
    reported = tuple(int(details[key]) for key in ("Размер окна, точек", "Горизонт h, шагов", "Шаг данных, с"))
    if reported != expected:
        parser.error("Window, horizon or frequency in the report does not match the model")
    normalization = checkpoint["normalization"]
    columns = normalization["columns"]
    if gru_table["Параметр"].duplicated().any() or set(gru_table["Параметр"]) != set(columns):
        parser.error("Report variables do not match model variables")
    gru_table = gru_table.set_index("Параметр").loc[columns].reset_index()
    if not np.isfinite(gru_table["MAE"]).all() or (gru_table["MAE"] < 0).any():
        parser.error("Report contains invalid MAE values")

    frame = load_frame(data_path, normalization, config["frequency_seconds"])
    dataset = WindowDataset(frame, columns, config["window"], config["horizon"], config["frequency_seconds"])
    counts = (len(frame), frame["run_id"].nunique(), len(dataset))
    if counts != tuple(int(details[key]) for key in ("Строк оценки", "Прогонов", "Прогнозов")):
        parser.error("Data and forecast counts do not match the GRU report")
    scale = np.array([normalization["scale"][c] for c in columns])
    table = comparison_table(gru_table, baseline_mae(dataset, scale))
    improvement = table["Улучшение MAE"]
    better, worse, equal = (int((improvement > 0).sum()), int((improvement < 0).sum()), int((improvement == 0).sum()))
    comparison_details = [
        ("Отчёт GRU", relative_path(args.report)),
        *display_details(details),
        ("GRU лучше (переменных)", better),
        ("GRU хуже (переменных)", worse),
        ("Равная MAE (переменных)", equal),
        ("Среднее изменение по группам", average_change_summary(table)),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([f"{name}: {value}"] for name, value in comparison_details)
        writer.writerow([])
        table.to_csv(file, index=False, float_format="%.10g", na_rep="—")
    print(f"Comparison: {args.output}; GRU better: {better}, worse: {worse}, equal: {equal}")


if __name__ == "__main__":
    main()
