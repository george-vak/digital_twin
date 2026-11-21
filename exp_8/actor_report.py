"""Compare actor-only, hybrid and built-in control on paired simulator seeds."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def traces(path):
    result = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            result[int(row['seed'])].append(row)
    return result


def report(directory):
    selection = json.loads((directory / 'selection.json').read_text())
    with (directory / 'comparison.csv').open() as stream:
        candidates = list(csv.DictReader(stream))
    paths = {'Актор': directory / 'evaluation', 'rebuilt': ROOT / 'exp_8/runs/rebuilt/evaluation',
             'Штатный регулятор': directory / 'baseline_evaluation'}
    data = {name: traces(path / 'steps.csv') for name, path in paths.items()}
    seeds = list(data['Актор'])
    if any(set(values) != set(seeds) for values in data.values()):
        raise ValueError('Comparison requires identical test seeds')
    if not all(row['intervention'] == 'True' for rows in data['Актор'].values() for row in rows):
        raise ValueError('Actor-only evaluation contains a non-actor step')
    horizon = min(len(rows) for values in data.values() for rows in values.values())
    lines = ['# Самостоятельное управление актором', '',
             f"По валидации выбран `{selection['candidate']}`.", '',
             'Эпизоды начинаются после штатного прогрева. На всех шагах оценки актора MV выдаёт нейросеть.',
             '', '## Выбор на валидации', '',
             '| Вариант | Эпизодов ≥ 200 | Эпизодов по 300 | Средняя длина | Средняя стоимость |',
             '|---|---:|---:|---:|---:|']
    for row in candidates:
        lines.append(f"| {row['candidate']} | {row['episodes_at_least_200']} | {row['completed_episodes']} | "
                     f"{float(row['mean_steps']):.1f} | {float(row['mean_cost']):.2f} |")
    lines.extend(['', '## Оценка на новых seed', '',
                  f'Общий горизонт для всех политик и seed — {horizon} шагов.', '',
                  '| Политика | Эпизодов ≥ 200 | Эпизодов по 300 | Средняя длина | Стоимость за общий горизонт | CV[16] за общий горизонт |',
                  '|---|---:|---:|---:|---:|---:|'])
    for name, values in data.items():
        lengths = [len(rows) for rows in values.values()]
        completed = sum(rows[-1]['truncated'] == 'True' for rows in values.values())
        common = [row for rows in values.values() for row in rows[:horizon]]
        lines.append(f"| {name} | {sum(n >= 200 for n in lengths)}/{len(seeds)} | "
                     f"{completed}/{len(seeds)} | {mean(lengths):.1f} | "
                     f"{mean(float(r['cost']) for r in common):.2f} | "
                     f"{mean(float(r['CV[16]']) for r in common):.2f} |")
    long_seeds = [seed for seed in seeds if all(len(values[seed]) >= 200 for values in data.values())]
    if long_seeds:
        lines.extend(['', f'На {len(long_seeds)} общих seed все три политики прошли первые 200 шагов. '
                      'Средняя стоимость за эти 200 шагов приведена ниже.', ''])
        for name, values in data.items():
            cohort = [float(row['cost']) for seed in long_seeds for row in values[seed][:200]]
            lines.append(f'- {name} — {mean(cohort):.2f}.')
    if all(len(data[name][seed]) >= 200 for seed in seeds for name in ('Актор', 'Штатный регулятор')):
        lines.extend(['', 'Первые 200 шагов актора и штатного управления на всех итоговых seed.', '',
                      '| Политика | Средняя стоимость | Средний CV[16] | Шаги внутри всех CV-границ |',
                      '|---|---:|---:|---:|'])
        for name in ('Актор', 'Штатный регулятор'):
            rows = [row for seed in seeds for row in data[name][seed][:200]]
            safe = sum(row['safe_step'] == 'True' for row in rows)
            lines.append(f"| {name} | {mean(float(row['cost']) for row in rows):.2f} | "
                         f"{mean(float(row['CV[16]']) for row in rows):.2f} | {safe}/{len(rows)} |")
    lines.extend(['', 'CV[16] показывает расход выходного потока. Для оценки экономии на единицу '
                  'годного продукта дополнительно нужны его количество и качество.', '',
                  '| Seed | Шагов у актора | Стоимость актора | Стоимость rebuilt | Стоимость штатного регулятора |',
                  '|---|---:|---:|---:|---:|'])
    for seed in seeds:
        size = min(len(values[seed]) for values in data.values())
        costs = [mean(float(row['cost']) for row in values[seed][:size]) for values in data.values()]
        lines.append(f"| {seed} | {len(data['Актор'][seed])} | " + ' | '.join(f'{cost:.2f}' for cost in costs) + ' |')
    lines.extend(['', 'Для сравнения на каждом seed использовано одинаковое число шагов среди трёх политик.', ''])
    (directory / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='actor_anchored')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('Invalid run name')
    report(ROOT / 'exp_8/runs' / args.run)
