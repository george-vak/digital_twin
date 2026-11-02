import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def plot_run(run_dir, episode=None, phase="train"):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    frame = pd.read_csv(run_dir / "steps.csv")

    train = frame.loc[frame.phase == phase].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError("No observations for selected phase in steps.csv")
    train["global_step"] = np.arange(1, len(train) + 1)
    groups = list(train.groupby("episode", sort=False))
    selected = groups[-1][0] if episode is None else episode
    detail = train.loc[train.episode == selected]
    if detail.empty:
        raise ValueError(f"Episode {selected} is absent; available: {[key for key, _ in groups]}")
    output = run_dir / "plots"
    output.mkdir(exist_ok=True)
    paths = []

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    for index, (_, group) in enumerate(groups):
        axes[0].plot(group.global_step, group.cost, color="#2563eb", lw=1.5,
                     label="Фактическая стоимость" if index == 0 else None)
        axes[0].axvline(group.global_step.iloc[-1] + .5, color="gray", alpha=.25, lw=.8)
    axes[0].set(title="Стоимость по ходу обучения" if phase == "train" else "Стоимость при оценке",
                xlabel="Шаг взаимодействия с симулятором", ylabel="Стоимость Analyser")
    if "released" in train:
        released = train.released.astype(str).str.lower().eq("true")
        axes[0].scatter(train.loc[released, "global_step"], train.loc[released, "cost"],
                        s=8, color="#d97706", label="Внутреннее управление")
    first_update = train.loc[train.actor_loss.notna(), "global_step"]
    if not first_update.empty:
        axes[0].axvline(first_update.iloc[0], color="#f59e0b", ls="--", label="Начало обновлений актёра")
    axes[0].legend()
    numbers = np.arange(1, len(groups) + 1)
    means = [group.cost.mean() for _, group in groups]
    stds = [group.cost.std(ddof=0) for _, group in groups]
    axes[1].errorbar(numbers, means, yerr=stds, fmt="o-", capsize=4, color="#2563eb")
    axes[1].set(title="Стоимость каждого эпизода: среднее ± std", ylabel="Стоимость Analyser")
    stopped = [str(group.terminated.iloc[-1]).lower() == "true" for _, group in groups]
    axes[2].bar(numbers, [len(group) for _, group in groups], color=["#dc2626" if stop else "#059669" for stop in stopped])
    axes[2].axhline(config["episode_steps"], color="gray", ls="--", label="Лимит эпизода")
    axes[2].set(title="Длительность",
                xlabel="Номер эпизода", ylabel="Шаги")
    axes[2].legend()
    for ax in axes[1:]:
        ax.set_xticks(numbers)
    fig.suptitle(f"", fontsize=15)
    save(fig, "01_training.png" if phase == "train" else "01_evaluation.png")

    minutes = detail.step * config["frequency_seconds"] / 60
    for prefix, count, columns, name in (("CV", 41, 4, "03_cv"), ("MV", 12, 3, "04_mv")):
        variables = [f"{prefix}[{i}]" for i in range(count)]
        available = [column for column in variables if column in detail and detail[column].notna().any()]
        if not available:
            continue
        rows = math.ceil(len(available) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(15, 2.2 * rows), squeeze=False,
                                 sharex=True, layout="constrained")
        for ax, column in zip(axes.flat, available):
            ax.plot(minutes, detail[column], color="#2563eb" if prefix == "CV" else "#059669", lw=1.3)
            if (run_dir / "bounds.json").exists():
                bounds = json.loads((run_dir / "bounds.json").read_text())
                j = int(column.split("[")[1][:-1]) + (41 if prefix == "MV" else 0)
                center = bounds["center"][j]
                for key, color in (("p80_radius", "#d97706"), ("p90_radius", "#dc2626")):
                    radius = bounds[key][j]
                    ax.axhline(center-radius, color=color, ls="--", lw=.7)
                    ax.axhline(center+radius, color=color, ls="--", lw=.7)
            ax.set_title(column)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        for ax in list(axes.flat)[len(available):]:
            ax.set_visible(False)
        fig.supxlabel("Минуты управления после прогрева")
        fig.supylabel("Физические единицы переменных" if prefix == "CV" else "Команда MV, %")
        fig.suptitle(f"Порог: P{config.get('threshold_percentile', 90)} · P80: оранжевый · P90: красный · {prefix} · эпизод {int(selected) + 1} (id={selected}) · {len(detail)} шагов", fontsize=15)
        save(fig, f"{name}_episode_{int(selected):03d}.png")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Name under exp_5/runs")
    parser.add_argument("--episode", type=int, help="Zero-based episode id; default: last")
    parser.add_argument("--phase", choices=("train", "validation", "evaluation"), default="train")
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("--run must be a directory name")
    run_dir = ROOT / "exp_5" / "runs" / args.run
    if args.phase != "train":
        run_dir = run_dir / args.phase
    for path in plot_run(run_dir, args.episode, args.phase):
        print(path)


if __name__ == "__main__":
    main()
