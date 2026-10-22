import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from eval import data_fingerprint, file_sha256
from experiment_paths import relative_path
from generate import fit_normalization
from model import GRUModel, WindowDataset, load_physical_frame
from preprocess import difference_frame


def mase_scale(train, columns, level_scale):
    delta = difference_frame(train, columns)
    return delta[columns].abs().mean().to_numpy() / level_scale


def scaled_error_statistics(errors, denominator):
    """Errors are standardized level errors; zero scale stays undefined."""
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    scaled = np.full_like(absolute, np.nan)
    np.divide(absolute, denominator, out=scaled, where=np.asarray(denominator) > 0)
    return scaled.mean(axis=0), scaled.std(axis=0, ddof=0), absolute.mean(axis=0)


def write_results_table(table, results):
    results.mkdir(parents=True, exist_ok=True)
    details = results / ".metadata"
    details.mkdir(exist_ok=True)
    # Retain temporal std and persistence metrics outside the compact user table.
    (details / "mase_details.json").write_text(
        table.to_json(orient="records", force_ascii=False, indent=2, double_precision=15) + "\n",
        encoding="utf-8",
    )
    compact = table[["variable", "exp1_MASE", "exp2_MASE", "improvement_percent"]].rename(
        columns={"exp1_MASE": "MASE1", "exp2_MASE": "MASE2", "improvement_percent": "imp_percent"})
    compact.to_csv(results / "mase_comparison.csv", index=False, float_format="%.10g", na_rep="NA")


@torch.inference_mode()
def predict_levels(checkpoint, levels):
    """Predict all valid targets and index predictions by (run_id, step)."""
    config, normalization = checkpoint["config"], checkpoint["normalization"]
    if config["horizon"] != 1 or config["frequency_seconds"] != 60:
        raise ValueError("MASE comparison requires h=1 and a 60-second frequency")
    columns = normalization["columns"]
    transform = config.get("target_transform", "levels")
    if transform not in ("levels", "first_difference"):
        raise ValueError(f"Unsupported transform: {transform}")
    frame = difference_frame(levels, columns) if transform == "first_difference" else levels.copy()
    mean = np.array([normalization["mean"][c] for c in columns])
    scale = np.array([normalization["scale"][c] for c in columns])
    frame[columns] = (frame[columns] - mean) / scale
    dataset = WindowDataset(frame, columns, config["window"], 1, 60)
    model = GRUModel(config["input_size"], config["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    outputs = np.concatenate([model(x).numpy().astype(np.float64) for x, _ in DataLoader(dataset, batch_size=256)])
    outputs = outputs * scale + mean
    first_target = config["window"] + (transform == "first_difference")
    keys, bases = [], []
    for _, run in levels.groupby("run_id", sort=False):
        run = run.sort_values("step")
        keys.append(run.iloc[first_target:][["run_id", "step"]])
        bases.append(run.iloc[first_target - 1:-1][columns].to_numpy())
    if transform == "first_difference":
        # Only the last observed level enters the reconstruction, never the target.
        outputs += np.concatenate(bases)
    index = pd.MultiIndex.from_frame(pd.concat(keys, ignore_index=True))
    if index.has_duplicates or not np.isfinite(outputs).all():
        raise ValueError("Invalid forecast keys or non-finite forecasts")
    return pd.DataFrame(outputs, index=index, columns=columns)


def write_report(table, summary, output):
    valid = table["exp1_MASE"].notna() & table["exp2_MASE"].notna()
    lines = [
        "ЭКСПЕРИМЕНТ №2: ДИФФЕРЕНЦИРОВАНИЕ И СТАНДАРТИЗАЦИЯ",
        "",
        "Задача: прогноз 53 переменных (41 CV + 12 MV) на 1 минуту вперёд.",
        "Штатный режим симулятора; шаг 60 с. Данные 20 000/5000/5000:",
        "train — seeds 1–4, validation — seed 6, test — seed 7.",
        "Архитектура обоих вариантов: один GRU(53,64) и линейный выход 64→53.",
        "exp_1 использует сохранённые веса, обученные на уровнях; exp_2 —",
        "одно обучение с seed 42 на первых разностях. Повторов обучения нет.",
        "Разности вычисляются внутри прогонов, затем центрируются и делятся",
        "на std_train (ddof=0). Mean/std рассчитаны только по train.",
        "Вход exp_2 — 60 разностей (61 исходное наблюдение); exp_1 — 60 уровней.",
        "Обучение: MSE на стандартизованных разностях, Adam, lr=0.001, batch=256,",
        "до 30 эпох, early stopping=5; выбирается минимум validation MSE.",
        f"Лучшая эпоха exp_2: {summary['exp2_best_epoch']}; validation MSE разностей: {summary['exp2_validation_mse']:.6f}.",
        "Прогноз уровня восстанавливается из последнего известного уровня",
        "и предсказанной разности после отмены её стандартизации.",
        "",
        f"Оценка: {summary['common_targets']} общих тестовых точек, шаги 61…4999.",
        "Ошибка и общий знаменатель выражены в шкале стандартизованных уровней.",
        "MASE_j = mean_test(|z_hat_j-z_j|) / mean_train(|z_t,j-z_(t-1),j|).",
        "Знаменатель общий для exp_1, exp_2 и persistence; межпрогонные переходы",
        "в него не входят. Persistence предсказывает последнее наблюдение.",
        "MASE ± std — среднее и std (ddof=0) абсолютных масштабированных ошибок",
        "по тестовым точкам каждой переменной. Это не доверительный интервал",
        "и не оценка разброса повторных обучений.",
        "",
        f"По MASE exp_2 лучше exp_1: {summary['exp2_better']} переменных; хуже: {summary['exp2_worse']}; равны: {summary['equal']}.",
        f"MASE определена для {int(valid.sum())} из {len(table)} переменных.",
    ]
    for label in ("exp1", "exp2", "persistence"):
        lines.append(f"Средняя MASE по определённым переменным, {label}: {summary['macro_mase'][label]:.6f}.")
    lines += ["Эти сводные средние дают всем определённым переменным одинаковый вес.", "",
              "Примеры (mean ± std по времени):", "Переменная | exp_1 | exp_2 | persistence"]
    for variable in ("CV[1]", "CV[6]", "CV[8]", "CV[37]"):
        row = table.set_index("variable").loc[variable]
        values = [f"{row[label + '_MASE']:.4f} ± {row[label + '_std']:.4f}" for label in ("exp1", "exp2", "persistence")]
        lines.append(variable + " | " + " | ".join(values))
    lines += ["", "Переменные с нулевым знаменателем не включены в MASE и её агрегаты."]
    for _, row in table[~valid].iterrows():
        lines.append(f"{row['variable']}: MAE в стандартизованной шкале, exp_1={row['exp1_MAE_z']:.6g}, exp_2={row['exp2_MAE_z']:.6g}; MASE не определена.")
    lines += ["", "Вывод: сравнение характеризует эффект обучения на разностях при",
              "неизменной архитектуре на одном тестовом прогоне штатного режима.",
              "Статистическая значимость и вариативность обучения не оценивались.",
              "Одинаковые целевые точки обеспечивают сопоставимость ошибок; объём",
              "исходной истории отличается на одну точку из-за дифференцирования.",
              "Таблица MASE1, MASE2, imp_percent: results/mase_comparison.csv.",
              "Подробные метрики: results/.metadata/mase_details.json. Прогнозы и общий train-",
              "знаменатель сохранены рядом, параметры и хеши — в results/summary.json."]
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    defined = table.dropna(subset=["exp1_MASE", "exp2_MASE"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    ax = axes[0]
    ax.scatter(defined["exp1_MASE"], defined["exp2_MASE"], color="#2876b9")
    limit = max(defined["exp1_MASE"].max(), defined["exp2_MASE"].max()) * 1.05
    ax.plot([0, limit], [0, limit], "--", color="#777777")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="MASE exp_1", ylabel="MASE exp_2",
           title="Каждая точка — переменная\nНиже диагонали: exp_2 лучше")
    ax.grid(alpha=0.2)
    changes = defined.sort_values("improvement_percent")
    ax = axes[1]
    colors = np.where(changes["improvement_percent"] >= 0, "#269d80", "#dc695e")
    ax.barh(changes["variable"], changes["improvement_percent"], color=colors)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.tick_params(axis="y", labelsize=6)
    ax.set(xlabel="Снижение MASE exp_2 относительно exp_1, %", title="Положительное изменение: exp_2 лучше")
    fig.suptitle("h=1 мин, общие точки test seed 7; одинаковый train-знаменатель")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "01_mase_comparison.png", dpi=180)
    plt.close(fig)


def plot_mase2(table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(24, 7), sharey=True, layout="constrained")
    positions = np.arange(len(table))
    colors = np.array(["#2876b9" if name.startswith("CV") else "#269d80" for name in table["variable"]])
    for ax, metric, title in zip(axes, ("exp2_MASE", "persistence_MASE"),
                                 ("MASE exp_2", "MASE persistence — последнее значение")):
        values = table[metric].to_numpy()
        defined = np.isfinite(values)
        ax.bar(positions[defined], values[defined], color=colors[defined], width=0.75)
        reference = ax.axhline(1, color="#d62728", linestyle="--", linewidth=1.8,
                              label="MASE = 1 (наивный прогноз на train)")
        for position in positions[~defined]:
            ax.text(position, 0.02, "NA", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color="#777777")
        ax.set(xticks=positions, xticklabels=table["variable"], xlabel="Переменная",
               ylabel="MASE", ylim=(0, None), xlim=(-0.75, len(table) - 0.25), title=title)
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(handles=[Patch(color="#2876b9", label="CV"), Patch(color="#269d80", label="MV"), reference])
    fig.suptitle("h=1 мин · общие тестовые точки и шкала Y · NA: нулевой train-знаменатель")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "02_mase2_by_variable.png", dpi=180)
    plt.close(fig)


def evaluate(data, reference_path, model_path, output, skip_report=False):
    torch.set_num_threads(4)
    level_normalization = json.loads((data / "normalization.json").read_text())
    columns = level_normalization["columns"]
    level_scale = np.array([level_normalization["scale"][c] for c in columns])
    train = load_physical_frame(data / "train.csv.gz", columns, 60)
    test = load_physical_frame(data / "test.csv.gz", columns, 60)
    models = {"exp1": torch.load(reference_path, map_location="cpu", weights_only=True),
              "exp2": torch.load(model_path, map_location="cpu", weights_only=True)}
    expected = {"exp1": level_normalization,
                "exp2": fit_normalization(difference_frame(train, columns), columns)}
    for label, checkpoint in models.items():
        transform = checkpoint["config"].get("target_transform", "levels")
        if transform != {"exp1": "levels", "exp2": "first_difference"}[label]:
            raise ValueError(f"Unexpected preprocessing for {label}: {transform}")
        if checkpoint["normalization"]["columns"] != columns:
            raise ValueError(f"Variable mismatch: {label}")
        for key in ("mean", "scale"):
            if not np.allclose([checkpoint["normalization"][key][c] for c in columns],
                               [expected[label][key][c] for c in columns], rtol=1e-8, atol=1e-10):
                raise ValueError(f"{label}: normalization does not match training data")
    for key in ("input_size", "hidden_size", "window", "horizon", "frequency_seconds"):
        if models["exp1"]["config"][key] != models["exp2"]["config"][key]:
            raise ValueError(f"Different model configurations: {key}")
    predictions = {label: predict_levels(checkpoint, test) for label, checkpoint in models.items()}
    common = predictions["exp1"].index.intersection(predictions["exp2"].index, sort=False)
    if not len(common):
        raise ValueError("No common forecast targets")
    actual = test.set_index(["run_id", "step"])[columns].loc[common]
    ordered = pd.concat([run.sort_values("step") for _, run in test.groupby("run_id", sort=False)])
    baseline = ordered.copy()
    baseline[columns] = ordered.groupby("run_id", sort=False)[columns].shift(1)
    predictions["persistence"] = baseline.set_index(["run_id", "step"])[columns]
    denominator = mase_scale(train, columns, level_scale)
    variables = pd.read_csv(data / "variables.csv", index_col="variable").loc[columns]
    table = variables[["description", "unit"]].reset_index()
    table["n_test"] = len(common)
    table["train_denominator_z"] = denominator
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    for label, prediction in predictions.items():
        prediction = prediction.loc[common]
        errors = (prediction.to_numpy() - actual.to_numpy()) / level_scale
        if not np.isfinite(errors).all():
            raise ValueError("Non-finite forecast errors")
        mean, std, mae_z = scaled_error_statistics(errors, denominator)
        table[f"{label}_MASE"], table[f"{label}_std"], table[f"{label}_MAE_z"] = mean, std, mae_z
        prediction.to_csv(results / f"predictions_{label}_levels.csv.gz", float_format="%.17g")
    actual.to_csv(results / "actual_levels.csv.gz", float_format="%.17g")
    table["improvement_percent"] = 100 * (table["exp1_MASE"] - table["exp2_MASE"]) / table["exp1_MASE"].replace(0, np.nan)
    write_results_table(table, results)
    valid = denominator > 0
    difference = table.loc[valid, "exp1_MASE"] - table.loc[valid, "exp2_MASE"]
    summary = {
        "frequency_seconds": 60, "horizon": 1, "common_targets": len(common), "std_ddof": 0,
        "mean_std_axis": "test points within each variable, absolute scaled errors",
        "train_pairs": sum(len(run) - 1 for _, run in train.groupby("run_id")),
        "zero_denominator_variables": [c for c, value in zip(columns, denominator) if value == 0],
        "exp2_better": int((difference > 0).sum()), "exp2_worse": int((difference < 0).sum()),
        "equal": int((difference == 0).sum()),
        "macro_mase": {label: float(table.loc[valid, f"{label}_MASE"].mean()) for label in predictions},
        "exp2_best_epoch": models["exp2"]["training"]["best_epoch"],
        "exp2_validation_mse": models["exp2"]["training"]["validation_mse"],
        "files": {"data": relative_path(data.resolve()), "exp1_model": relative_path(reference_path.resolve()), "exp2_model": relative_path(model_path.resolve())},
        "sha256": {"train": data_fingerprint(data / "train.csv.gz"), "test": data_fingerprint(data / "test.csv.gz"),
                   "exp1_model": file_sha256(reference_path), "exp2_model": file_sha256(model_path)},
        "evaluation_normalization": level_normalization,
        "train_denominator_z": dict(zip(columns, denominator.tolist())),
    }
    (results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if not skip_report:
        write_report(table, summary, output / "reports")
    plot_comparison(table, output / "plots")
    plot_mase2(table, output / "plots")
    print(f"Common targets: {len(common)}; exp_2 better/worse/equal: {summary['exp2_better']}/{summary['exp2_worse']}/{summary['equal']}")
    print(f"Macro MASE (defined variables only): {summary['macro_mase']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-report", action="store_true", help="Keep the existing text report unchanged")
    args = parser.parse_args()
    evaluate(args.data, args.reference_model, args.model, args.output_dir, skip_report=args.skip_report)
