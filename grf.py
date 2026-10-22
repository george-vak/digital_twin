import argparse
import csv
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from experiment_paths import DEFAULT_RUN, ROOT, resolve_path


GRU, BASE = "MAE GRU", "MAE baseline"


def read_report(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    start = next((i for i, row in enumerate(rows) if row and row[0] == "Параметр"), None)
    if start is None:
        return None
    details = dict(row[0].split(": ", 1) for row in rows[:start] if row and ": " in row[0])
    table = pd.DataFrame([r for r in rows[start + 1:] if r], columns=rows[start])
    table = table.rename(columns={"MAE": GRU}).set_index("Параметр")
    for column in (GRU, BASE):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="raise")
            if not np.isfinite(table[column]).all() or (table[column] < 0).any():
                raise ValueError(f"{path}: некорректная MAE")
    if table.index.has_duplicates:
        raise ValueError(f"{path}: повторяющиеся переменные")
    data = Path(details["Файл оценки"])
    return {
        "dataset": f"{data.parent.name}/{data.name}",
        "h": float(details["Горизонт (с)"]) / 60,
        "model": Path(details["Модель"]).name,
        "model_path": resolve_path(details["Модель"]),
        "data_path": resolve_path(data),
        "window": details["Размер окна (точек)"],
        "table": table,
        "path": path,
    }


def load_reports(folders):
    reports = {}
    for folder in folders:
        for path in sorted(folder.glob("*.csv")):
            report = read_report(path)
            if report is None:
                continue
            key = (report["dataset"], report["h"], report["model"], report["window"])
            previous = reports.get(key)
            if previous is None or (BASE in report["table"] and BASE not in previous["table"]):
                reports[key] = report
    return sorted(reports.values(), key=lambda r: (r["dataset"], r["h"], r["model"]))


def label(report):
    return f"{report['h']:g} мин\n{report['model']}"


def comparison_plots(reports):
    comparisons = [r for r in reports if BASE in r["table"]]
    datasets = sorted({r["dataset"] for r in comparisons})
    if not datasets:
        return []
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5),
                             squeeze=False, layout="constrained")
    for ax, dataset in zip(axes[0], datasets):
        runs = [r for r in comparisons if r["dataset"] == dataset]
        bottom = np.zeros(len(runs))
        for name, color, sign in (("GRU лучше", "#269d80", 1),
                                  ("GRU хуже", "#dc695e", -1),
                                  ("Равная MAE", "#adb5bd", 0)):
            counts = [int((np.sign(r["table"][BASE] - r["table"][GRU]) == sign).sum()) for r in runs]
            bars = ax.bar(range(len(runs)), counts, bottom=bottom, label=name, color=color)
            ax.bar_label(bars, labels=[str(n) if n else "" for n in counts], label_type="center")
            bottom += counts
        ax.set(title=dataset, ylabel="Число переменных", xticks=range(len(runs)),
               xticklabels=[label(r) for r in runs])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.suptitle("По скольким переменным GRU превосходит прогноз последним значением")

    # Единая шкала для обоих тестов; нулевая MAE эталона не даёт процента.
    changes = [(r["table"][BASE] - r["table"][GRU]) / r["table"][BASE].replace(0, np.nan) * 100
               for r in comparisons]
    finite = np.concatenate([s.dropna().to_numpy() for s in changes])
    norm = TwoSlopeNorm(vmin=min(-1, finite.min()) if finite.size else -1,
                        vcenter=0, vmax=max(1, finite.max()) if finite.size else 1)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#dedede")
    variables = list(dict.fromkeys(v for r in comparisons for v in r["table"].index))
    heat, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), max(6, len(variables) * 0.27)),
                              squeeze=False, layout="constrained")
    for ax, dataset in zip(axes[0], datasets):
        indices = [i for i, r in enumerate(comparisons) if r["dataset"] == dataset]
        matrix = np.column_stack([changes[i].reindex(variables).to_numpy() for i in indices])
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        ax.set(title=dataset, xticks=range(len(indices)),
               xticklabels=[label(comparisons[i]) for i in indices],
               yticks=range(len(variables)), yticklabels=variables)
        for row, col in np.ndindex(matrix.shape):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.1f}" if np.isfinite(value) else "—",
                    ha="center", va="center", fontsize=8)
    heat.colorbar(im, ax=list(axes[0]), label="Снижение MAE относительно эталона, %", shrink=0.65)
    heat.suptitle("Зелёный: GRU лучше; красный: хуже\nСерый: процент не определён или данные отсутствуют")
    return [("01_comparison_counts.png", fig), ("02_improvement_heatmap.png", heat)]


def gru_residuals(report):
    """Остатки в исходных единицах на тех же окнах, что использует eval.py."""
    try:
        import torch
    except ImportError as error:
        raise ValueError("Для расчёта остатков нужен PyTorch: python3 -m pip install -r requirements.txt") from error
    from torch.utils.data import DataLoader
    from model import GRUModel, WindowDataset, load_frame

    torch.set_num_threads(4)
    checkpoint = torch.load(report["model_path"], map_location="cpu", weights_only=True)
    config, normalization = checkpoint["config"], checkpoint["normalization"]
    if (config["horizon"] != 1 or config["frequency_seconds"] != 60
            or config["window"] != int(report["window"])):
        raise ValueError("Параметры модели не соответствуют отчёту для h=1 мин")
    columns = normalization["columns"]
    model = GRUModel(config["input_size"], config["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    frame = load_frame(report["data_path"], normalization, config["frequency_seconds"])
    dataset = WindowDataset(frame, columns, config["window"], 1, config["frequency_seconds"])
    scale = np.array([normalization["scale"][c] for c in columns])
    with torch.inference_mode():
        errors = [(model(x).numpy().astype(np.float64) - y.numpy()) * scale
                  for x, y in DataLoader(dataset, batch_size=256)]
    residuals = pd.DataFrame(np.concatenate(errors), columns=columns)
    if not np.isfinite(residuals.to_numpy()).all():
        raise ValueError("Остатки содержат NaN или бесконечные значения")
    # Не смешиваем старые отчёты с результатами изменённой модели/данных.
    if not np.allclose(residuals.abs().mean(), report["table"].loc[columns, GRU],
                       rtol=1e-4, atol=1e-7):
        raise ValueError(f"{report['path']}: пересчитанная MAE не совпадает с отчётом")
    print(f"Остатки {report['dataset']}: {len(residuals)} прогнозов; MAE совпадает с отчётом.")
    return residuals


def residual_plots(reports):
    selected = [r for r in reports if r["h"] == 1]
    if not selected:
        print("Нет отчётов для h=1 мин: гистограммы остатков пропущены.")
    for report in selected:
        residuals = gru_residuals(report)
        rows = (len(residuals.columns) + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(19, rows * 2.7),
                                 squeeze=False, layout="constrained")
        for ax, variable in zip(axes.flat, residuals.columns):
            values = residuals[variable]
            ax.hist(values, bins=40, color="#2876b9", alpha=0.8)
            ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
            ax.axvline(values.mean(), color="#d55e00", linewidth=1.4)
            info = report["table"].loc[variable]
            unit = {"oC": "°C", "%": "п.п.", "mole %": "п.п. мольной доли"}.get(info["Единица"], info["Единица"])
            ax.set(title=f"{variable}: {fill(info['Описание'], 38)}",
                   xlabel=f"{unit}", ylabel="Число прогнозов")
            ax.text(0.98, 0.95, f"Среднее = {values.mean():.3g}\nMAE = {values.abs().mean():.3g}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.2)
        for ax in list(axes.flat)[len(residuals.columns):]:
            ax.set_visible(False)
        fig.suptitle(f"Остатки GRU, h=1 мин • {report['dataset']} • N={len(residuals)}\n"
                     "Шкалы переменных различаются.", fontsize=14)
        data = Path(report["dataset"])
        name = f"03_residuals_h1_{data.parent.name}_{data.name.removesuffix('.gz').removesuffix('.csv')}.png"
        yield name, fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN, help="Папка exp_i/runs/<run>")
    parser.add_argument("--input", type=Path, nargs="+", help="Папки с отчётами (по умолчанию comp и results выбранного запуска)")
    parser.add_argument("--output", type=Path, help="Папка для PNG (по умолчанию <run-dir>/plots)")
    parser.add_argument("--no-show", action="store_true", help="Сохранить PNG без открытия окон")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.run_dir / "plots"
    if args.no_show:
        plt.switch_backend("Agg")
    try:
        reports = load_reports(args.input or [args.run_dir / "comp", args.run_dir / "results"])
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if not reports:
        parser.error("Не найдены CSV-отчёты с таблицей MAE")
    # Каждый тест и горизонт должны быть представлены одной моделью.
    keys = [(r["dataset"], r["h"]) for r in reports]
    if len(keys) != len(set(keys)):
        parser.error("Несколько моделей на один горизонт: укажите --input с нужными отчётами")
    try:
        figures = comparison_plots(reports) + list(residual_plots(reports))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        path = args.output / name
        fig.savefig(path, dpi=180)
        print(path)
    # Удаляем только заменённый график, созданный прежней версией скрипта.
    (args.output / "03_mae_by_horizon.png").unlink(missing_ok=True)
    if not any(BASE in r["table"] for r in reports):
        print("В отчётах нет MAE baseline: построены только гистограммы остатков GRU.")
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
