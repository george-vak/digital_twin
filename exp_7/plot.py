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
    summary = json.loads((run_dir / "summary.json").read_text())
    episodes = summary['episodes']

    train = frame.loc[frame.phase == phase].copy().reset_index(drop=True)
    train["global_step"] = np.arange(1, len(train) + 1)
    groups = list(train.groupby("episode", sort=False))
    selected = len(episodes)-1 if episode is None else episode
    if selected < 0 or selected >= len(episodes):
        raise ValueError(f'Episode {selected} is absent')
    detail = train.loc[train.episode == selected]
    if detail.empty:
        detail = pd.read_csv(run_dir/'warmup'/f'episode_{int(selected):03d}.csv').tail(1).copy()
        detail['step'] = 0
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
        axes[0].plot(group.global_step, group.cost, color="#2563eb", lw=1.5, marker=".",
                     label="Фактическая стоимость" if index == 0 else None)
        axes[0].axvline(group.global_step.iloc[-1] + .5, color="gray", alpha=.25, lw=.8)
    axes[0].set(title="Стоимость по ходу обучения" if phase == "train" else "Стоимость при оценке",
                xlabel="Шаг взаимодействия с симулятором", ylabel="Стоимость Analyser")
    if "released" in train and train.released.astype(str).str.lower().eq("true").any():
        released = train.released.astype(str).str.lower().eq("true")
        axes[0].scatter(train.loc[released, "global_step"], train.loc[released, "cost"],
                        s=8, color="#d97706", label="Внутреннее управление")
    first_update = train.loc[train.actor_loss.notna(), "global_step"]
    if not first_update.empty:
        update_label = "Обновления после шагов среды" if config.get('actor_pretrain_updates',0) else "Начало обновлений актёра"
        axes[0].axvline(first_update.iloc[0], color="#f59e0b", ls="--", label=update_label)
    if 'target_cost' in config:
        axes[0].axhline(config['target_cost'], color="#dc2626", ls="--",
                        label=f"Целевая стоимость {config['target_cost']:g}")
    axes[0].legend()
    numbers = np.arange(1, len(episodes) + 1)
    means = [e['mean_cost'] if e['mean_cost'] is not None else np.nan for e in episodes]
    stds = [e['std_cost'] if e['std_cost'] is not None else np.nan for e in episodes]
    axes[1].errorbar(numbers, means, yerr=stds, fmt="o-", capsize=4, color="#2563eb")
    axes[1].set(title="Стоимость каждого эпизода: среднее ± std", ylabel="Стоимость Analyser")
    if 'target_cost' in config:
        axes[1].axhline(config['target_cost'], color="#dc2626", ls="--")
    stopped = [e['terminated'] for e in episodes]
    bars = axes[2].bar(numbers, [e['steps'] for e in episodes], color=["#dc2626" if stop else "#059669" for stop in stopped])
    axes[2].bar_label(bars, padding=3)
    if 'safe_steps' in episodes[0]:
        axes[2].scatter(numbers, [e['safe_steps'] for e in episodes], color='#2563eb', label='Безопасные шаги', zorder=3)
        axes[2].scatter(numbers, [e['safe_target_steps'] for e in episodes], color='#d97706', marker='x', label='Безопасные шаги при 160 ± 5', zorder=4)
    axes[2].axhline(config["episode_steps"], color="gray", ls="--", label="Лимит эпизода")
    axes[2].set(title="Длительность",
                xlabel="Номер эпизода", ylabel="Шаги")
    axes[2].legend()
    for ax in axes[1:]:
        ax.set_xticks(numbers)
    save(fig, "01_training.png" if phase == "train" else "01_evaluation.png")

    minutes = detail.step * config["frequency_seconds"] / 60
    for prefix, count, columns, name in (("CV", 41, 4, "03_cv"), ("MV", 11, 3, "04_mv")):
        variables = [f"{prefix}[{i}]" for i in range(count)]
        available = [column for column in variables if column in detail and detail[column].notna().any()]
        if not available:
            continue
        rows = math.ceil(len(available) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(15, 2.2 * rows), squeeze=False,
                                 sharex=True, layout="constrained")
        for ax, column in zip(axes.flat, available):
            ax.plot(minutes, detail[column], color="#2563eb" if prefix == "CV" else "#059669", lw=1.3, marker=".")
            bounds_path = run_dir / "warmup" / f"episode_{int(selected):03d}_bounds.json"
            if bounds_path.exists():
                bounds = json.loads(bounds_path.read_text())
                j = int(column.split("[")[1][:-1]) + (41 if prefix == "MV" else 0)
                if j < 52:
                    ax.axhline(bounds["lower"][j], color="#dc2626", ls="--", lw=.8)
                    ax.axhline(bounds["upper"][j], color="#dc2626", ls="--", lw=.8)
            ax.set_title(column)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        for ax in list(axes.flat)[len(available):]:
            ax.set_visible(False)
        fig.supxlabel("Минуты управления после прогрева")
        fig.supylabel("Физические единицы переменных" if prefix == "CV" else "Команда MV, %")
        boundary_label = "P1/P99 прогрева" if prefix == "CV" else "0–100%"
        caption = f"{episodes[int(selected)]['steps']} шагов"
        if episodes[int(selected)]['steps'] == 0:
            caption += ' · остановка после прогрева, показано начальное состояние'
        fig.suptitle(f"Границы {boundary_label}: красный · {prefix} · эпизод {int(selected) + 1} · {caption}", fontsize=13)
        save(fig, f"{name}_episode_{int(selected):03d}.png")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=".", help="Name under exp_7/runs; default: current run directly in runs")
    parser.add_argument("--episode", type=int, help="Zero-based episode id; default: last")
    parser.add_argument("--phase", choices=("train", "validation", "evaluation"), default="train")
    args = parser.parse_args()
    if args.run != "." and (Path(args.run).name != args.run or args.run == ".."):
        parser.error("--run must be a directory name")
    run_dir = ROOT / "exp_7" / "runs" / args.run
    if args.phase != "train":
        run_dir = run_dir / args.phase
    for path in plot_run(run_dir, args.episode, args.phase):
        print(path)


if __name__ == "__main__":
    main()
