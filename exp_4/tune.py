"""Train a fixed regularization grid, select on validation, evaluate every candidate."""
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from exp_4.evaluate import evaluate
from exp_4.plot import plot_run


def main():
    torch.set_num_threads(4)
    experiment = ROOT / 'exp_4'
    study = experiment / 'studies/window60'
    study.mkdir(parents=True, exist_ok=True)
    base = json.loads((experiment / 'config.json').read_text())
    weights = [0, 1, 100, 10000, 1000000]
    protocol = dict(weights=weights, window=60, training_seeds=list(range(100, 110)),
                    validation_seeds=list(range(200, 205)), evaluation_seeds=list(range(300, 310)),
                    selection='Maximize validation mean_steps; then minimize mean episode cost; then weight',
                    episode_steps=200, model_seed=base['seed'])
    (study / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    for weight in weights:
        name = f'tuning_w60_l{weight}'
        run_dir = experiment / 'runs' / name
        config = {**base, 'regularization_window': 60, 'regularization_weight': float(weight)}
        config_path = study / f'config_l{weight}.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        if not (run_dir / 'controller.pt').exists():
            with (study / f'train_l{weight}.log').open('w') as log:
                subprocess.run([sys.executable, str(experiment / 'run.py'), '--run', name,
                                '--config', str(config_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        else:
            if json.loads((run_dir / 'config.json').read_text()) != config:
                raise ValueError(f'Existing run config differs: {name}')
        print(f'Training complete: {name}', flush=True)
        summary_path = run_dir / 'validation/summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = evaluate(run_dir, protocol['validation_seeds'], phase='validation')
        rows.append(dict(run=name, weight=weight, mean_steps=summary['mean_steps'],
                         completed=summary['completed'], mean_cost=summary['mean_cost']))
        with (study / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Validation complete: {rows[-1]}', flush=True)
    selected = min(rows, key=lambda r: (-r['mean_steps'], r['mean_cost'], r['weight']))
    (study / 'selection.json').write_text(json.dumps(selected, indent=2) + '\n')
    print(f'Selected: {selected}', flush=True)
    # Evaluate every candidate on the same seeds; selection remains validation-only.
    for row in rows:
        run_dir = experiment / 'runs' / row['run']
        if not (run_dir / 'evaluation/summary.json').exists():
            evaluate(run_dir, protocol['evaluation_seeds'], all_plots=True)
        for phase in ('validation', 'evaluation'):
            output = run_dir / phase
            summary = json.loads((output / 'summary.json').read_text())
            for episode in range(len(summary['episodes'])):
                expected = [output / 'plots' / name for name in (
                    '01_evaluation.png', f'03_cv_episode_{episode:03d}.png',
                    f'04_mv_episode_{episode:03d}.png')]
                if not all(path.exists() for path in expected):
                    plot_run(output, episode=episode, phase=phase)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
    labels = [str(r['weight']) for r in rows]
    axes[0].bar(labels, [r['mean_steps'] for r in rows])
    axes[0].axhline(200, color='gray', linestyle='--')
    axes[0].set(ylabel='Шаги', title='Валидация: средняя длительность эпизода')
    axes[1].bar(labels, [r['mean_cost'] for r in rows])
    axes[1].set(ylabel='Стоимость Analyser', xlabel='Коэффициент регуляризации',
                title='Валидация: средняя стоимость (среднее по эпизодам)')
    fig.savefig(study / '01_parameter_selection.png', dpi=160)
    plt.close(fig)
    print(f'Study complete: {study}', flush=True)


if __name__ == '__main__':
    main()
