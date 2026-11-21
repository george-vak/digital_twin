"""Report forecast error, action ranking and paired policy evaluation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate(directory):
    def read(name):
        return json.loads((directory / f'{name}.json').read_text())

    forecast, critic = read('forecast'), read('critic')
    selection, result = read('selection'), read('result')
    settings = selection['settings']
    order = ('internal', 'nominal', 'actor')
    names = ('Встроенный регулятор', 'Нулевая поправка', 'Актор')
    seeds = settings['evaluation_seeds']
    for summary in result.values():
        if [row['seed'] for row in summary['episodes']] != seeds:
            raise ValueError('Evaluation seed groups differ')

    report = [
        '# exp_8. Удержание цены около 160', '',
        f"Цель — {settings['target_streak']} шагов подряд в пределах "
        f"{settings['target_cost']:g} ± {settings['tracking_tolerance']:g}.", '',
        '## Прогноз на валидации', '',
        '| GRU | RMSE цены +1 | RMSE цены +5 | RMSE цены +10 |',
        '|---|---:|---:|---:|',
    ]
    for model, metrics in forecast['validation'].items():
        report.append(f'| {model} | ' + ' | '.join(f"{metrics[str(h)]['price_rmse']:.3f}" for h in (1, 5, 10)) + ' |')
    report += ['', f"Выбрана модель `{forecast['selected']}` по сумме трёх ошибок прогноза цены.", '',
               '## Сравнение команд критиком на отложенных состояниях', '',
               '| Признаки Q | Верный порядок пар | RMSE разницы Q | Потеря при выборе команды |',
               '|---|---:|---:|---:|']
    for label, key in (('Исходная GRU', 'original_features'), ('Выбранная GRU', 'updated_features')):
        row = critic[key]
        report.append(f"| {label} | {row['pair_accuracy']:.1%} | {row['advantage_rmse']:.3f} | {row['regret']:.3f} |")
    row = critic['updated_features']
    report += ['', f"Критик обучен на {row['train_anchors']} состояниях. Оценка выполнена по {row['pairs']} "
               f"парам команд из {row['validation_anchors']} новых состояний. Средняя потеря при выборе "
               f"номинальной команды составляет {row['nominal_regret']:.3f}. Потеря равна разнице между "
               'лучшим наблюдаемым возвратом ветки и возвратом выбранной ветки.', '',
               '## Выбор актора на валидации', '',
               '| Кандидат | Скорость обучения | Период поправки | Предел поправки, п. п. | Вес отклонения от номинала | Эпизод обучения | Медиана серии | Средняя серия |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    for candidate in selection['candidates']:
        option = candidate['settings']
        limit = settings['residual_limit'] * option.get('correction_gain', 1.0)
        report.append(f"| {candidate['index']} | {option['actor_lr']:g} | {option['intervention_period']} | "
                      f"{limit:g} | {option['residual_weight']:g} | {candidate['epoch']} | "
                      f"{candidate['median_streak']:g} | {candidate['mean_streak']:.1f} |")
    report += ['', f"Выбран кандидат {selection['index']}, блок обучения №{selection['epoch']}. "
               'Хеши GRU и Q-критика проверены до и после оптимизации актора.', '',
               '## Итоговая оценка', '',
               '| Политика | Эпизодов ≥ 200 шагов | Серий ≥ 200 | Медиана серии | Максимум серии | MAE цены |',
               '|---|---:|---:|---:|---:|---:|']
    for name, key in zip(names, order):
        row = result[key]
        report.append(f"| {name} | {row['episodes_at_least_200']}/{len(seeds)} | "
                      f"{row['successful_episodes']}/{len(seeds)} | {row['median_streak']:g} | "
                      f"{row['max_streak']} | {row['cost_mae']:.2f} |")
    report += ['', 'Шаги сравниваются на одних и тех же seed. Нулевая поправка подаёт внешнюю '
               'команду, рассчитанную встроенным регулятором, по тому же расписанию, что актор.', '',
               '| Seed | Серия встроенного регулятора | Серия нулевой поправки | Серия актора | Шагов у актора | RMSE серии актора | Std серии актора |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for internal, nominal, actor in zip(*(result[key]['episodes'] for key in order)):
        error, spread = actor['streak_rmse'], actor['streak_std']
        quality = f'{error:.2f} | {spread:.2f}' if error is not None else '— | —'
        report.append(f"| {actor['seed']} | {internal['longest_streak']} | {nominal['longest_streak']} | "
                      f"{actor['longest_streak']} | {actor['steps']} | {quality} |")
    report += ['', '## Вывод', '']
    if result['actor']['successful_episodes'] == 0:
        report.append('Цель в 200 последовательных шагов на итоговой выборке пока не достигнута.')
    if result['actor']['cost_mae'] > result['internal']['cost_mae']:
        report.append('Средняя абсолютная ошибка цены у актора выше, чем у встроенного регулятора.')
    if (result['actor']['median_streak'] < result['nominal']['median_streak']
            and result['actor']['cost_mae'] > result['nominal']['cost_mae']):
        report.append('Выученная поправка проиграла нулевой по медиане длины серии и средней ошибке цены при том же расписании.')
    report += ['', 'Q-критик проверен на разовых изменениях команды с дальнейшим встроенным управлением. '
               'Длительная стабильность при повторных вмешательствах оценивается фактическими эпизодами.', '',
               '![Длина серии по каждому seed](streaks.png)', '',
               '![Выбор блока обучения актора](learning.png)', '']
    (directory / 'report.md').write_text('\n'.join(report))

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for name, key, offset in zip(names, order, (-0.25, 0, 0.25)):
        bars = ax.bar([index + offset for index in range(len(seeds))],
                      [row['longest_streak'] for row in result[key]['episodes']], width=0.24, label=name)
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.axhline(settings['target_streak'], ls='--', color='gray', label='Цель')
    ax.set(xticks=range(len(seeds)), xticklabels=seeds, xlabel='Seed', ylabel='Шаги подряд',
           title=f"Самая длинная серия при {settings['target_cost']:g} ± {settings['tracking_tolerance']:g}")
    ax.legend()
    fig.savefig(directory / 'streaks.png', dpi=160)
    plt.close(fig)

    curve = pd.read_csv(directory / f"candidate_{selection['index']:02d}/validation_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(curve.epoch, curve.median_streak, 'o-', label='Актор')
    axes[0].axhline(selection['baseline']['median_streak'], ls='--', color='gray', label=names[0])
    axes[0].set(ylabel='Медиана длины серии', title='Удержание цены на валидации')
    axes[0].legend()
    axes[1].plot(curve.epoch, curve.cost_mae, 'o-', color='#b45309')
    axes[1].set(ylabel='MAE цены', title='Ошибка цены на валидации')
    for axis in axes:
        axis.axvline(selection['epoch'], ls=':', color='#059669')
        axis.set(xlabel='Блок обучения', xticks=curve.epoch)
    fig.savefig(directory / 'learning.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='rebuilt')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('Invalid run name')
    generate(ROOT / 'exp_8/runs' / args.run)
