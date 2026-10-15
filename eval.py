import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import DEFAULT_DATA, ROOT, GRUModel, WindowDataset, load_frame


def file_sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def data_fingerprint(path):
    """Include sidecars: changing normalization also changes physical values."""
    paths = [path]
    metadata_path = path.parent / "metadata.json"
    if metadata_path.exists():
        paths.append(metadata_path)
        preprocessing = json.loads(metadata_path.read_text()).get("preprocessing", {})
        if isinstance(preprocessing, dict) and "parameters_file" in preprocessing:
            paths.append(path.parent / preprocessing["parameters_file"])
    return hashlib.sha256("\n".join(file_sha256(p) for p in paths).encode()).hexdigest()



def relative_path(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def report_metadata_path(path):
    return path.parent / ".metadata" / f"{path.name}.json"


def display_details(details):
    window = int(details["Размер окна, точек"])
    return [
        ("Модель", relative_path(details["Модель"])),
        ("Архитектура", details["Архитектура"].replace(", ", "; ")),
        ("Датасет обучения", relative_path(details["Датасет обучения"])),
        ("Строк обучения", details["Строк обучения"]),
        ("Строк валидации", details["Строк валидации"]),
        ("Лучшая эпоха", details["Лучшая эпоха"]),
        ("Окно", f"{window} точек от t−{window - 1} до t"),
        ("Размер окна (точек)", window),
        ("Шаг данных (с)", details["Шаг данных, с"]),
        ("Горизонт h (шагов)", details["Горизонт h, шагов"]),
        ("Горизонт (с)", details["Горизонт, с"]),
        ("Файл оценки", relative_path(details["Файл оценки"])),
        ("Строк оценки", details["Строк оценки"]),
        ("Прогонов (seeds)", details["Прогонов"]),
        ("Прогнозов", details["Прогнозов"]),
    ]


def write_evaluation_report(path, details, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([f"{name}: {value}"] for name, value in display_details(details))
        writer.writerow([])
        table.to_csv(file, index=False, float_format="%.10g")
    # Keep machine-readable verification data outside the human-readable CSV.
    metadata = report_metadata_path(path)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps({
        "details": {key: str(value) for key, value in details.items()},
        "report_sha256": file_sha256(path),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, default=DEFAULT_DATA / "test.csv.gz")
    parser.add_argument("--model", type=Path, required=True, help="Path to the saved model checkpoint")
    parser.add_argument("--output", type=Path, help="Override automatic CSV filename in results/")
    args = parser.parse_args()
    if args.output is None:
        file_name = args.file.name.removesuffix(".gz").removesuffix(".csv")
        dataset_name = args.file.resolve().parent.name
        args.output = ROOT / "results" / f"{args.model.stem}_{dataset_name}_{file_name}.csv"

    torch.set_num_threads(4)
    data_hash = data_fingerprint(args.file)
    model_hash = file_sha256(args.model)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    normalization = checkpoint["normalization"]
    columns = normalization["columns"]
    model = GRUModel(config["input_size"], config["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    frame = load_frame(args.file, normalization, config["frequency_seconds"])
    dataset = WindowDataset(frame, columns, config["window"], config["horizon"], config["frequency_seconds"])
    errors = np.zeros(len(columns), dtype=np.float64)
    scale = np.array([normalization["scale"][column] for column in columns])
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=256):
            # Difference in original units; the mean cancels in subtraction.
            errors += (np.abs(model(x).numpy().astype(np.float64) - y.numpy()) * scale).sum(axis=0)
    mae = errors / len(dataset)
    table = pd.DataFrame({
        "Параметр": columns,
        "Описание": [checkpoint["variables"][c]["description"] for c in columns],
        "Единица": [checkpoint["variables"][c]["unit"] for c in columns],
        "MAE": mae,
    })
    training = checkpoint["training"]
    details = [
        ("Модель", str(args.model.resolve())),
        ("SHA256 модели", model_hash),
        ("Архитектура", f"Один слой GRU, {config['hidden_size']} скрытых компонент; линейный выход на {len(columns)} переменных"),
        ("Датасет обучения", training["dataset"]),
        ("Строк обучения", training["rows"]["train"]),
        ("Строк валидации", training["rows"]["validation"]),
        ("Лучшая эпоха", training["best_epoch"]),
        ("Окно", f"{config['window']} точек: t−{config['window'] - 1}, …, t"),
        ("Размер окна, точек", config["window"]),
        ("Шаг данных, с", config["frequency_seconds"]),
        ("Горизонт h, шагов", config["horizon"]),
        ("Горизонт, с", config["horizon"] * config["frequency_seconds"]),
        ("Прогноз", "Одна точка t+h; каждое окно получает фактическую историю, прогнозы не подставляются в следующие окна"),
        ("Файл оценки", str(args.file.resolve())),
        ("SHA256 данных", data_hash),
        ("Строк оценки", len(frame)),
        ("Прогонов", frame["run_id"].nunique()),
        ("Прогнозов", len(dataset)),
        ("Границы окон", "Окна не пересекают границы прогонов"),
        ("Метрика", "MAE — средняя абсолютная ошибка по всем прогнозам для каждой переменной в исходных физических единицах"),
        ("Нормализация", "Параметры модели рассчитаны только по обучающей выборке"),
    ]
    write_evaluation_report(args.output, dict(details), table)
    print(f"Report: {args.output}; {len(dataset)} forecasts, h={config['horizon']}")


if __name__ == "__main__":
    main()
