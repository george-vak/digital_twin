"""Plot episode duration, episode prices and every CV trajectory in an exp_8 run."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('train', 'validation', 'evaluation', 'baseline_validation', 'baseline_evaluation', 'nominal_evaluation')
CV_COLUMNS = [f'CV[{index}]' for index in range(41)]


def plot_phase(directory, summary_only=False):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    frame = pd.read_csv(directory / 'steps.csv')
    episodes = summary['episodes']
    if not episodes:
        raise ValueError(f'No episodes in {directory}')
    output = directory / 'plots'
    output.mkdir(exist_ok=True)
    target = config['target_cost']
    tolerance = config['tracking_tolerance']
    indices = np.arange(1, len(episodes) + 1)
    seeds = [episode['seed'] for episode in episodes]
    colors = ['#059669' if episode['completed'] else '#dc2626' for episode in episodes]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')
    for index, episode in enumerate(episodes):
        trace = frame.loc[frame.seed == episode['seed']]
        axes[0].plot(trace.step, trace.cost, lw=1.2, alpha=0.8,
                     label=f"Эпизод {index + 1} · seed {episode['seed']}")
    axes[0].axhline(target, color='black', ls='--', lw=1, label=f'Цель {target:g}')
    axes[0].axhspan(target - tolerance, target + tolerance, color='#059669', alpha=0.08)
    axes[0].set(title='Стоимость по шагам каждого эпизода', xlabel='Шаг после прогрева', ylabel='Стоимость')
    axes[0].legend(ncol=min(3, len(episodes)), fontsize=8)

    means = np.array([episode['mean_cost'] for episode in episodes], dtype=float)
    errors = np.array([episode['cost_mae'] for episode in episodes], dtype=float)
    deviations = [frame.loc[frame.seed == seed, 'cost'].std(ddof=0) for seed in seeds]
    axes[1].errorbar(indices, means, yerr=deviations, fmt='o-', capsize=4,
                     color='#2563eb', label='Средняя стоимость ± std')
    for index, mean, error in zip(indices, means, errors):
        axes[1].annotate(f'MAE {error:.1f}', (index, mean), xytext=(0, 10),
                         textcoords='offset points', ha='center', fontsize=8, color='#b45309')
    axes[1].axhline(target, color='black', ls='--', lw=1)
    axes[1].set(title='Стоимость по эпизодам', ylabel='Стоимость')
    axes[1].legend(fontsize=8)

    bars = axes[2].bar(indices, [episode['steps'] for episode in episodes], color=colors)
    axes[2].bar_label(bars, padding=3, fontsize=8)
    if 'longest_streak' in episodes[0]:
        axes[2].scatter(indices, [episode['longest_streak'] for episode in episodes], marker='D',
                        color='#2563eb', zorder=3, label='Самая длинная серия около 160')
        axes[2].legend(fontsize=8, loc='upper left')
    observed_max = max(episode['steps'] for episode in episodes)
    if config['episode_steps'] <= observed_max * 1.25:
        axes[2].axhline(config['episode_steps'], color='gray', ls='--', label='Лимит эпизода')
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.99, 0.98, f"Лимит эпизода — {config['episode_steps']} шагов",
                     transform=axes[2].transAxes, ha='right', va='top', color='gray')
    axes[2].set(title='Количество шагов по эпизодам', xlabel='Эпизод', ylabel='Шаги',
                ylim=(0, observed_max * 1.25))
    for axis in axes[1:]:
        axis.set_xticks(indices, [f'{index}\n{seed}' for index, seed in zip(indices, seeds)])
    axes[2].set_xlabel('Эпизод / seed')
    path = output / '01_episodes.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths = [path]
    if summary_only:
        return paths

    for number, episode in enumerate(episodes, start=1):
        seed = episode['seed']
        trace = frame.loc[frame.seed == seed].sort_values('step')
        if len(trace) != episode['steps']:
            raise ValueError(f'Episode {seed} has inconsistent step count')
        bounds = json.loads((directory / f'warmup_{seed}_bounds.json').read_text())
        warmup = pd.read_csv(directory / f'warmup_{seed}.csv').tail(1)
        time = np.r_[0, trace.step.to_numpy(dtype=int)]
        fig, axes = plt.subplots(math.ceil(41 / 4), 4, figsize=(16, 24),
                                 sharex=True, squeeze=False, layout='constrained')
        for index, column in enumerate(CV_COLUMNS):
            axis = axes.flat[index]
            values = np.r_[warmup[column].iloc[0], trace[column].to_numpy(dtype=float)]
            axis.plot(time, values, color='#2563eb', lw=1.3)
            axis.axhline(bounds['lower'][index], color='#dc2626', ls='--', lw=0.8)
            axis.axhline(bounds['upper'][index], color='#dc2626', ls='--', lw=0.8)
            axis.set_title(column)
            axis.ticklabel_format(axis='y', style='plain', useOffset=False)
        for axis in list(axes.flat)[41:]:
            axis.set_visible(False)
        fig.supxlabel('Минуты после прогрева, 0 — последнее наблюдение прогрева')
        fig.supylabel('Значение CV в физических единицах')
        fig.suptitle(f'Эпизод {number} · seed {seed} · {episode["steps"]} шагов · красный — P1/P99 прогрева',
                     fontsize=15)
        path = output / f'03_cv_episode_{number - 1:03d}_seed_{seed}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='rebuilt', help='Name under exp_8/runs')
    parser.add_argument('--phase', choices=(*PHASES, 'all'), default='all')
    parser.add_argument('--summary-only', action='store_true', help='Regenerate only the episode summary')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('Invalid run name')
    root = ROOT / 'exp_8/runs' / args.run
    selection_path = root / 'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    for phase in PHASES if args.phase == 'all' else (args.phase,):
        if phase == 'train' and 'index' in selection:
            directory = root / f"candidate_{selection['index']:02d}"
        elif phase == 'validation' and 'candidate' in selection:
            directory = root / selection['candidate']
        else:
            directory = root if phase == 'train' else root / phase
        if args.phase == 'all' and not (directory / 'summary.json').exists():
            continue
        for path in plot_phase(directory, summary_only=args.summary_only):
            print(path)


if __name__ == '__main__':
    main()
