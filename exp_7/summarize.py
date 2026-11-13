"""Validate the current run and regenerate its metrics and short conclusion."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from exp_7.run import aggregate
from exp_7.safety import cv_violations


def summarize(run_name=None):
    root = Path(__file__).resolve().parent
    if run_name is None:
        current = root/'current_run.json'
        run_name = json.loads(current.read_text())['run'] if current.exists() else '.'
    run = root/'runs'/run_name
    reports = []
    for phase in ('train','validation','evaluation'):
        directory = run if phase == 'train' else run/phase
        config = json.loads((directory/'config.json').read_text())
        count = len(config['seeds'])
        frame = pd.read_csv(directory/'steps.csv')
        summary = json.loads((directory/'summary.json').read_text())
        assert config['actor_objective'] == 'finite_horizon_safe_tracking_return'
        assert len(summary['episodes']) == count
        assert len(frame) == summary['total_steps']
        assert np.isfinite(frame[['cost', *[f'CV[{j}]' for j in range(41)],
                                  *[f'MV[{j}]' for j in range(12)]]].to_numpy(dtype=float)).all()
        assert not frame.released.any()
        reasons = []
        for episode, item in enumerate(summary['episodes']):
            group = frame.loc[frame.episode == episode]
            bounds = json.loads((directory/'warmup'/f'episode_{episode:03d}_bounds.json').read_text())
            warmup = pd.read_csv(directory/'warmup'/f'episode_{episode:03d}.csv')
            np.testing.assert_allclose(bounds['lower'][:41], np.quantile(warmup.iloc[:,:41], .01, axis=0))
            np.testing.assert_allclose(bounds['upper'][:41], np.quantile(warmup.iloc[:,:41], .99, axis=0))
            np.testing.assert_array_equal(bounds['lower'][41:], np.zeros(11))
            np.testing.assert_array_equal(bounds['upper'][41:], np.full(11,100.))
            initial = cv_violations(warmup.iloc[-1].to_numpy(), bounds)
            assert initial == item['initial_cv_violation']
            assert len(group) == item['steps'] <= config['episode_steps']
            if initial:
                assert group.empty and item['terminated'] and not item['truncated']
                assert item['cost_mae'] is None and item['mean_cost'] is None
                assert item['safe_steps'] == item['safe_target_steps'] == 0
                assert item['termination_reason'] == initial
            else:
                assert not group.empty
                np.testing.assert_array_equal(group.step, np.arange(1,len(group)+1))
                safe_count = target_count = streak = longest = 0
                for i, (_, row) in enumerate(group.iterrows()):
                    violation = cv_violations(row[[f'CV[{j}]' for j in range(41)]].to_numpy(dtype=float), bounds)
                    assert violation == ('' if pd.isna(row.cv_violation) else row.cv_violation)
                    if i < len(group)-1:
                        assert not row.terminated and not row.truncated and not violation
                    else:
                        assert row.terminated or row.truncated
                        assert bool(row.terminated) == item['terminated']
                        assert bool(row.truncated) == item['truncated']
                    if violation:
                        assert row.terminated and not row.truncated
                    safe = not row.terminated
                    on_target = safe and abs(row.cost-config['target_cost']) <= config['tracking_tolerance']
                    assert safe == row.safe_step and on_target == row.safe_target_step
                    expected_reward = float(safe)*(config['survival_weight']+config['tracking_weight']*
                                                   np.exp(-((row.cost-config['target_cost'])/config['cost_scale'])**2))
                    np.testing.assert_allclose(row.reward, expected_reward)
                    safe_count += safe
                    target_count += on_target
                    streak = streak+1 if on_target else 0
                    longest = max(longest,streak)
                    np.testing.assert_allclose(row[[f'MV[{j}]' for j in range(11)]].to_numpy(dtype=float),
                                               row[[f'candidate_MV[{j}]' for j in range(11)]].to_numpy(dtype=float))
                    np.testing.assert_allclose(row['MV[11]'],bounds['fixed_mv11'])
                assert item['safe_steps'] == safe_count and item['safe_target_steps'] == target_count
                assert item['longest_target_streak'] == longest
                if item['truncated']:
                    assert len(group) == 200
                np.testing.assert_allclose(item['cost_mae'], np.abs(group.cost-160).mean())
                np.testing.assert_allclose(item['cost_rmse'], np.sqrt(((group.cost-160)**2).mean()))
            reasons.extend(filter(None,item['termination_reason'].split(';')))
        expected = aggregate(summary['episodes'], summary['actor_updates'], config)
        assert all(summary[key] == value for key,value in expected.items())
        if phase == 'train':
            offline = config.get('actor_pretrain_updates',0)
            online = len(frame)*config.get('updates_per_step',1)
            assert summary['actor_updates'] == offline+online
            if 'model_updates' in summary:
                assert summary['online_actor_updates'] == online
                assert summary['pretrain_actor_updates'] == offline
                assert summary['pretrain_model_updates'] == config.get('model_pretrain_updates',0)+offline
                assert summary['model_updates'] == config.get('model_pretrain_updates',0)+offline+online
            if (directory/'pretrain.csv').exists():
                pretrain = pd.read_csv(directory/'pretrain.csv')
                assert len(pretrain) == config.get('model_pretrain_updates',0)+offline
                joint = pretrain.loc[pretrain.stage == 'joint']
                assert len(joint) == offline
                np.testing.assert_allclose(joint.actor_loss,joint.actor_survival_loss+
                    joint.actor_tracking_objective+joint.actor_regularization+joint.actor_barrier,rtol=1e-5,atol=1e-6)
                assert np.isfinite(pretrain.model_loss).all()
            assert np.isfinite(frame[['model_loss','actor_loss', 'actor_expected_safe_steps']].to_numpy()).all()
            np.testing.assert_allclose(frame.actor_loss, frame.actor_survival_loss+frame.actor_tracking_objective+
                                       frame.actor_regularization+frame.actor_barrier, rtol=1e-5)
            assert frame.actor_expected_safe_steps.between(0,200).all()
            assert (frame.actor_tracking_return <= frame.actor_expected_safe_steps+1e-6).all()
        else:
            assert summary['actor_updates'] == 0
        plots = list((directory/'plots').glob('*.png'))
        assert len(plots) == 1+count*2
        for path in plots:
            with Image.open(path) as image:
                image.verify()
        reports.append(dict(phase=phase, **{key:value for key,value in summary.items() if key != 'episodes'},
                            episode_count=count, plots=len(plots),
                            budget_fraction=len(frame)/(count*config['episode_steps']),
                            cv_terminated=sum('CV[' in e['termination_reason'] for e in summary['episodes']),
                            top_terminal_reasons=pd.Series(reasons).value_counts().head(5).to_dict()))
    result = dict(run=run_name,target_cost=160,episode_steps=200,tracking_tolerance=config['tracking_tolerance'],
                  actor_objective=config['actor_objective'],phases=reports)
    (root/'results_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    training_config = json.loads((run/'config.json').read_text())
    lines = ['# Краткий вывод по эксперименту 7', '',
        'Цель эксперимента — как можно дольше удерживать цену около 160 при соблюдении всех границ CV, до 200 минутных шагов. '
        'Границы P1/P99 вычисляются по 1000 шагам прогрева каждого эпизода и затем фиксируются. '
        'Нарушение завершает эпизод, включая случай недопустимого состояния сразу после прогрева.', '',
        'По сравнению с exp_6 добавили накопленную награду за безопасную длительность и близость цены к 160 на прогнозной траектории. '
        'При выходе за барьер управление заканчивается. В exp_6 внутренний регулятор обеспечивал большую часть 200-шаговых эпизодов, '
        'актёр управлял 16 из 2000 оценочных шагов.', '']
    if training_config.get('actor_pretrain_updates',0):
        lines.append(f"Перед взаимодействием с симулятором использовали сохранённые переходы exp_4. "
                     f"Число предварительных обновлений только модели динамики — {training_config.get('model_pretrain_updates',0)}, "
                     f"совместных обновлений актёра и динамики — {training_config['actor_pretrain_updates']}. "
                     f"Число совместных обновлений после каждого нового перехода — {training_config.get('updates_per_step',1)}.")
        lines.append('')
    study = root/'reports'
    if run_name == '.' and (study/'selection.json').exists():
        selection = json.loads((study/'selection.json').read_text())
        lines += ['Провели совместный поиск по 24 наборам гиперпараметров. Три лучших варианта проверили с двумя seed обучения '
                  'и бюджетами 10 и 30 эпизодов. Параметры выбирали по validation, сначала по безопасному времени при цене 160 ± 5, '
                  'затем по общей безопасной длительности. Финальную модель обучили с отдельным seed 45 и оценили на новых seed 900–919.', '',
                  f"Выбран вариант {selection['name']}. Learning rate актёра {training_config['actor_lr']:g}, модели динамики "
                  f"{training_config['model_lr']:g}, batch {training_config['batch_size']}, шум {training_config['exploration_std']:g}. "
                  f"Масштаб ошибки цены {training_config['cost_scale']:g}, вес близости цены {training_config['tracking_weight']:g}, "
                  f"вес барьера {training_config['barrier_weight']:g}, вес регуляризации {training_config['regularization_weight']:g} "
                  f"при окне {training_config['regularization_window']} шагов.", '']
    lines += ['| Этап | Эпизодов | Выполнено шагов | Безопасных шагов | Безопасных шагов при 160 ± 5 | До 200 шагов | MAE к 160 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    labels = dict(train='Обучение',validation='Валидация',evaluation='Оценка')
    for report in reports:
        mae = '—' if report['cost_mae'] is None else f"{report['cost_mae']:.2f}"
        safe = round(report['mean_safe_steps']*report['episode_count'])
        lines.append(f"| {labels[report['phase']]} | {report['episode_count']} | {report['total_steps']} | {safe} | "
                     f"{report['safe_target_steps']} | {report['completed_episodes']} | {mae} |")
    evaluation = reports[-1]
    lines += ['',
        f"На итоговой оценке {evaluation['initial_cv_outside']} из {evaluation['episode_count']} эпизодов завершились уже после прогрева. "
        f"До 200 шагов дошли {evaluation['completed_episodes']} эпизодов. Средняя безопасная длительность составила "
        f"{evaluation['mean_safe_steps']:.2f} шага на эпизод. Обучение дало {reports[0]['actor_updates']} обновлений актёра.", '',
        'Хорошо, что настройки теперь проверены на нескольких вариантах обучения, а финальная оценка использует отдельные начальные условия. '
        'Качество управления оценивается одновременно по безопасности и цене. Основные ограничения — частые недопустимые стартовые состояния, '
        'небольшое разнообразие безопасных переходов и накопление ошибки многошагового прогноза. '
        'Достигнутый результат следует оценивать по приведённым безопасным шагам и эпизодам до 200.', '',
        'Следующий этап — проверить соответствие статистических границ технологическим ограничениям, увеличить набор безопасных переходов '
        'и проверить модель динамики на длинных траекториях. Выполненный поиск ограничен указанными диапазонами гиперпараметров.', '',
        f"Текущий запуск сохранён в `{('runs' if run_name == '.' else 'runs/'+run_name)}`. Проверка CV относится к минутным наблюдениям. "
        'Стоимость и ошибки рассчитаны по выполненным переходам, включая последний шаг с нарушением. '
        'Нулевые эпизоды учитываются в средней длительности.']
    (root/'краткий_вывод.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    args = parser.parse_args()
    if args.run not in (None,'.') and (Path(args.run).name != args.run or args.run == '..'):
        parser.error('Invalid run name')
    summarize(args.run)
