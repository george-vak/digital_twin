"""Reproducible validation-only hyperparameter search and held-out final evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from exp_3.simulator import DockerSimulator
from exp_7.data import Environment, Replay
from exp_7.networks import Transform
from exp_7.run import replay_signature, seed_replay
from exp_7.safety import cv_violations

STUDY = ROOT/'exp_7/reports'
DATA = ROOT/'exp_7/data'
SPACE = dict(
    actor_lr=[3e-6,3e-5,3e-4],
    model_lr=[1e-5,1e-4,3e-4],
    batch_size=[16,32,64],
    exploration_std=[0.,.1,.3],
    regularization_window=[10,30,60],
    regularization_weight=[0.,1.,100.,10000.],
    barrier_weight=[.1,1.,10.],
    barrier_epsilon=[.05,.1,.2],
    tracking_weight=[.3,1.,3.],
    cost_scale=[5.,10.,20.],
    survival_temperature=[.02,.05,.1],
    model_pretrain_updates=[0,32,96],
    actor_pretrain_updates=[32,96,192],
    updates_per_step=[1,4],
)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def prepare_replay(base):
    DATA.mkdir(parents=True,exist_ok=True)
    path = DATA/'parent_replay.npz'
    signature = replay_signature(base)
    if path.exists():
        with np.load(path,allow_pickle=False) as cache:
            if json.loads(str(cache['signature'])) != signature:
                raise ValueError('Existing replay cache has a different source')
        return path
    source = torch.load(ROOT/base['checkpoint'],map_location='cpu',weights_only=False)
    saved = torch.load(ROOT/base['parent_controller'],map_location='cpu',weights_only=False)
    transform = Transform(source)
    replay = Replay(base['buffer_size'],base['seed'],base['regularization_window'])
    backend = DockerSimulator(base['docker_image'])
    warmup = DATA/'parent_bounds'
    warmup.mkdir(exist_ok=True)
    try:
        env = Environment(backend,base,transform.window)
        seed_replay(env,replay,base,saved,warmup)
    finally:
        backend.close()
    arrays = {key:np.stack(items) for key,items in zip(
        ('levels','action','following','lower','upper','remaining'),zip(*replay.rows))}
    temporary = DATA/'parent_replay.tmp.npz'
    np.savez_compressed(temporary,**arrays,signature=json.dumps(signature))
    temporary.replace(path)
    safe = sum(not cv_violations(row[0][-1],dict(lower=row[3],upper=row[4])) for row in replay.rows)
    write_json(DATA/'replay_summary.json',dict(transitions=len(replay.rows),safe_start_transitions=safe,**signature))
    print(f'Prepared replay with {len(replay.rows)} transitions, {safe} safe starting states',flush=True)
    return path


def candidates(base, count=24):
    rng = np.random.default_rng(314159)
    balanced = {}
    for key, values in SPACE.items():
        sequence = (values*((count-2)//len(values)+1))[:count-1]
        rng.shuffle(sequence)
        balanced[key] = sequence
    configs = [dict(base,model_pretrain_updates=0,actor_pretrain_updates=0,updates_per_step=1)]
    for index in range(count-1):
        configs.append({**base,**{key:values[index] for key,values in balanced.items()}})
    return configs


def validation_metrics(directory):
    summary = json.loads((directory/'summary.json').read_text())
    frame = pd.read_csv(directory/'steps.csv')
    safe_costs = frame.loc[frame.safe_step.astype(str).str.lower().eq('true'),'cost']
    # Fixed scoring scale and tolerance are shared by all candidates.
    quality = float(np.exp(-((safe_costs-160.)/10.)**2).sum())
    count = len(summary['episodes'])
    return dict(mean_safe_target_steps=summary['safe_target_steps']/count,
                mean_safe_steps=summary['mean_safe_steps'],safe_price_score=quality/count,
                mean_steps=summary['mean_steps'],completed_episodes=summary['completed_episodes'],
                cost_mae=summary['cost_mae'],initial_cv_outside=summary['initial_cv_outside'],
                episodes=count)


def rank(row):
    mae = row['cost_mae'] if row['cost_mae'] is not None else float('inf')
    return (-row['mean_safe_target_steps'],-row['mean_safe_steps'],-row['safe_price_score'],mae,row['name'])


def execute_trial(name, config):
    directory = ROOT/'exp_7/runs'/name
    label = 'final' if name == '.' else name
    config_path = STUDY/'configs'/f'{label}.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f'Configuration changed for existing trial {name}')
    write_json(config_path,config)
    start = time.monotonic()
    for phase in ('train','validation'):
        output = directory if phase == 'train' else directory/phase
        complete = (output/'summary.json').exists()
        if phase == 'train':
            complete = complete and (output/'controller.pt').exists()
        if complete:
            recorded = json.loads((output/'config.json').read_text())
            if any(recorded.get(key) != value for key,value in config.items()):
                raise ValueError(f'Completed trial {name}/{phase} has a different configuration')
            continue
        if output.exists():
            if output == ROOT/'exp_7/runs':
                if any(not p.is_dir() or not p.name.startswith('tune_') for p in output.iterdir()):
                    raise FileExistsError('Existing final run is incomplete; refusing to overwrite it')
            else:
                output.rename(output.with_name(output.name+f'_incomplete_{time.time_ns()}'))
        command = [sys.executable,'-B','exp_7/run.py','--run',name,'--config',str(config_path),
                   '--phase',phase,'--no-plots','--threads','2']
        with (STUDY/'logs'/f'{label}_{phase}.log').open('w') as log:
            result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={
                **os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':'/tmp/exp7-matplotlib'})
        if result.returncode:
            return dict(name=name,status='failed',phase=phase,config=config,
                        elapsed_seconds=time.monotonic()-start)
    training = json.loads((directory/'summary.json').read_text())
    return dict(name=name,status='complete',config=config,**validation_metrics(directory/'validation'),
                actor_updates=training['actor_updates'],train_steps=training['total_steps'],
                elapsed_seconds=time.monotonic()-start)


def run_trials(trials, stage, workers):
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute_trial,name,config):name for name,config in trials}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(STUDY/f'{stage}.json',sorted(rows,key=lambda item:item['name']))
            message = {key:value for key,value in row.items() if key != 'config'}
            print(json.dumps(dict(stage=stage,**message),ensure_ascii=False),flush=True)
    return rows


def confirm_options(screening, workers):
    successful = sorted((row for row in screening if row['status']=='complete'),key=rank)
    if len(successful) < 3:
        raise RuntimeError('Fewer than three completed candidates')
    shortlist = successful[:3]
    write_json(STUDY/'shortlist.json',[dict(name=row['name'],config=row['config']) for row in shortlist])
    trials = []
    for position, row in enumerate(shortlist):
        for episodes in (10,30):
            for seed in (43,44):
                name = f'tune_confirm_{position:02d}_e{episodes}_s{seed}'
                config = dict(row['config'],episodes=episodes,seed=seed,validation_seeds=list(range(700,720)))
                trials.append((name,config))
    confirmed = run_trials(trials,'confirmation',workers)
    options = []
    for position, row in enumerate(shortlist):
        for episodes in (10,30):
            replicas = [item for item in confirmed if item['name'].startswith(f'tune_confirm_{position:02d}_e{episodes}_')]
            if len(replicas)!=2 or any(item['status']!='complete' for item in replicas):
                continue
            scores = {key:float(np.mean([item[key] for item in replicas])) for key in
                      ('mean_safe_target_steps','mean_safe_steps','safe_price_score','mean_steps','cost_mae')}
            options.append(dict(name=f'{row["name"]}_e{episodes}',config=dict(row['config'],episodes=episodes,
                                validation_seeds=list(range(700,720))),
                                replicas=[item['name'] for item in replicas],**scores))
    if not options:
        raise RuntimeError('No configuration completed both confirmation seeds')
    write_json(STUDY/'confirmed_options.json',sorted(options,key=rank))
    return min(options,key=rank)


def final_evaluation(selected, base):
    config = dict(selected['config'],seed=45,validation_seeds=list(range(700,720)),
                  evaluation_seeds=list(range(900,920)))
    write_json(STUDY/'selected_config.json',config)
    selection = dict(selected,final_run='.',final_model_seed=45,
                     evaluation_seeds=list(range(900,920)))
    write_json(STUDY/'selection.json',selection)
    print('Selection locked before final training and evaluation',flush=True)
    final = execute_trial('.',config)
    if final['status'] != 'complete':
        raise RuntimeError('Final model training failed')
    write_json(STUDY/'final_validation.json',final)
    directory = ROOT/'exp_7/runs'
    if not (directory/'evaluation/summary.json').exists():
        with (STUDY/'logs/final_evaluation.log').open('w') as log:
            subprocess.run([sys.executable,'-B','exp_7/run.py','--run','.','--phase','evaluation',
                            '--no-plots','--threads','2'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    checkpoint = directory/'controller.pt'
    write_json(STUDY/'final_result.json',dict(run='.',
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        validation=validation_metrics(directory/'validation'),evaluation=validation_metrics(directory/'evaluation')))
    # The selected configuration is now the normal training entry point.
    write_json(ROOT/'exp_7/config.json',config)
    write_json(ROOT/'exp_7/current_run.json',dict(run='.',study='reports'))
    print(json.dumps(json.loads((STUDY/'final_result.json').read_text()),ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error('workers must be 1 or 2')
    STUDY.mkdir(parents=True,exist_ok=True)
    for name in ('configs','logs'):
        (STUDY/name).mkdir(exist_ok=True)
    base_path = STUDY/'base_config.json'
    if base_path.exists():
        base = json.loads(base_path.read_text())
    else:
        base = json.loads((ROOT/'exp_7/config.json').read_text())
        write_json(base_path,base)
    torch.set_num_threads(2)
    cache = prepare_replay(base)
    if args.prepare_only:
        return
    base = dict(base,replay_cache=str(cache.relative_to(ROOT)),validation_seeds=list(range(700,710)),
                evaluation_seeds=list(range(900,920)))
    protocol = dict(search='balanced random joint search',search_seed=314159,candidates=24,space=SPACE,
                    screening=dict(train_seeds=list(range(600,610)),model_seed=42,validation_seeds=list(range(700,710))),
                    confirmation=dict(top_candidates=3,episode_budgets=[10,30],model_seeds=[43,44],
                                      validation_seeds=list(range(700,720))),
                    final=dict(model_seed=45,evaluation_seeds=list(range(900,920))),
                    selection=['maximize safe target steps per episode at 160 +/- 5',
                               'maximize mean safe steps','maximize safe price score with fixed scale 10',
                               'minimize observed cost MAE','candidate name for exact ties'],
                    fixed=dict(target_cost=160,episode_steps=200,warmup_steps=1000,survival_weight=1,
                               tracking_tolerance=5,buffer_size=10000,regularization_scale=100,survival_cutoff=1e-6),
                    baseline_reference='pre-tuning exp_7 model (removed during cleanup)')
    write_json(STUDY/'protocol.json',protocol)
    trials = [(f'tune_trial_{i:02d}',config) for i,config in enumerate(candidates(base))]
    screening = run_trials(trials,'screening',args.workers)
    selected = confirm_options(screening,args.workers)
    final_evaluation(selected,base)


if __name__ == '__main__':
    main()
