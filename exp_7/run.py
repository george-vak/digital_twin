"""Train finite-horizon price tracking with termination at the CV barrier."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp_7.networks import Actor, Dynamics, Transform, cost, adapt_state, actor_update
from exp_3.simulator import DockerSimulator
from exp_7.data import Environment, Replay
from exp_4.run import model_update, tensors, observation_fields
from exp_4.networks import validate_regularization
from exp_7.safety import cv_violations
from exp_7.plot import plot_run

METRICS = ['actor_loss', 'actor_cost', 'actor_tracking_loss', 'actor_expected_safe_steps',
           'actor_tracking_return', 'actor_survival_loss', 'actor_tracking_objective',
           'actor_penalty', 'actor_regularization', 'cv_barrier', 'mv_barrier', 'actor_barrier']
COLUMNS = [f'CV[{i}]' for i in range(41)] + [f'MV[{i}]' for i in range(12)]


def validate_config(config, transform):
    if config.get('actor_objective') != 'finite_horizon_safe_tracking_return':
        raise ValueError('Expected finite_horizon_safe_tracking_return objective')
    if config.get('target_cost') != 160 or config.get('episode_steps') != 200:
        raise ValueError('exp_7 requires target_cost=160 and episode_steps=200')
    if config['frequency_seconds'] != transform.frequency:
        raise ValueError('Simulator and model frequencies must agree')
    validate_regularization(config, transform.window + 1)
    for key in ('cost_scale', 'survival_temperature', 'survival_weight', 'tracking_weight', 'tracking_tolerance'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0 < config['barrier_epsilon'] < 1 or not 0 < config['survival_cutoff'] < 1:
        raise ValueError('Barrier epsilon and survival cutoff must be between 0 and 1')
    if not np.isfinite(config['barrier_weight']) or config['barrier_weight'] < 0:
        raise ValueError('barrier_weight must be finite and nonnegative')
    for key in ('actor_lr', 'model_lr'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not np.isfinite(config['exploration_std']) or config['exploration_std'] < 0:
        raise ValueError('exploration_std must be finite and nonnegative')
    for key, default, minimum in (('model_pretrain_updates',0,0), ('actor_pretrain_updates',0,0),
                                  ('updates_per_step',1,1), ('batch_size',64,1), ('buffer_size',10000,1)):
        value = config.get(key, default)
        if isinstance(value,bool) or not isinstance(value,int) or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')


def summarize_episode(phase, seed, costs, safe_steps, target_steps, longest_streak, env, initial_violation):
    errors = np.asarray(costs) - env.config['target_cost']
    return dict(mode=phase, seed=seed, steps=len(costs), safe_steps=safe_steps,
                safe_target_steps=target_steps, longest_target_streak=longest_streak,
                terminated=bool(env.termination_reason), truncated=env.steps == env.config['episode_steps'] and not env.termination_reason,
                initial_cv_violation=initial_violation, termination_reason=env.termination_reason,
                mean_cost=float(np.mean(costs)) if costs else None,
                std_cost=float(np.std(costs)) if costs else None, total_cost=float(np.sum(costs)),
                target_cost=env.config['target_cost'],
                cost_mae=float(np.abs(errors).mean()) if costs else None,
                cost_rmse=float(np.sqrt(np.mean(errors**2))) if costs else None,
                released_steps=0, actor_steps=len(costs), release_fraction=0.)


def aggregate(summaries, updates, config):
    total = sum(s['steps'] for s in summaries)
    observed = [s for s in summaries if s['steps']]
    return dict(episodes=summaries, total_steps=total, actor_updates=updates,
                mean_steps=float(np.mean([s['steps'] for s in summaries])),
                mean_safe_steps=float(np.mean([s['safe_steps'] for s in summaries])),
                completed_episodes=sum(s['truncated'] for s in summaries),
                initial_cv_outside=sum(bool(s['initial_cv_violation']) for s in summaries),
                safe_target_steps=sum(s['safe_target_steps'] for s in summaries),
                target_cost=config['target_cost'], tracking_tolerance=config['tracking_tolerance'],
                mean_cost=sum(s['total_cost'] for s in observed)/total if total else None,
                cost_mae=sum(s['cost_mae']*s['steps'] for s in observed)/total if total else None,
                cost_rmse=float(np.sqrt(sum(s['cost_rmse']**2*s['steps'] for s in observed)/total)) if total else None,
                release_fraction=0.)


def replay_signature(config):
    return dict(replay_sha256=hashlib.sha256((ROOT/config['replay_source']).read_bytes()).hexdigest(),
                parent_sha256=hashlib.sha256((ROOT/config['parent_controller']).read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256((ROOT/config['checkpoint']).read_bytes()).hexdigest(),
                warmup_steps=config['warmup_steps'],frequency_seconds=config['frequency_seconds'],
                episode_steps=config['episode_steps'])


def seed_replay(env, replay, config, saved, warmup_dir):
    if config.get('replay_cache'):
        with np.load(ROOT/config['replay_cache'],allow_pickle=False) as cache:
            if json.loads(str(cache['signature'])) != replay_signature(config):
                raise ValueError('Replay cache does not match source data and configuration')
            for row in zip(*(cache[key] for key in ('levels','action','following','lower','upper','remaining'))):
                replay.add(row[0],row[1],row[2],dict(lower=row[3],upper=row[4]),row[5])
        return
    frame = pd.read_csv(ROOT/config['replay_source'])
    for episode, group in frame.groupby('episode', sort=False):
        env.reset(saved['config']['train_seed_start'] + int(episode))
        (warmup_dir/f'parent_episode_{int(episode):03d}_bounds.json').write_text(json.dumps(env.bounds, indent=2)+'\n')
        # Include the actual pre-action warmup history, so early safe states are available.
        values = np.concatenate((env.levels, group[COLUMNS].to_numpy(dtype=np.float64)))
        for step in range(min(len(group), config['episode_steps'])):
            index = env.window + step
            replay.add(values[index-env.window:index+1], values[index+1,41:52],
                       values[index+1], env.bounds, config['episode_steps']-step)


def update_networks(actor, model, transform, actor_opt, model_opt, replay, config):
    model_loss = model_update(model,transform,model_opt,replay.sample(config['batch_size'],transform))
    metrics = actor_update(actor,model,transform,actor_opt,
                           replay.sample(config['batch_size'],transform,safe_only=True),config)
    return model_loss, metrics


def pretrain(actor, model, transform, actor_opt, model_opt, replay, config, output):
    with (output/'pretrain.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['stage','update','model_loss',*METRICS])
        writer.writeheader()
        for update in range(config.get('model_pretrain_updates',0)):
            loss = model_update(model,transform,model_opt,replay.sample(config['batch_size'],transform))
            writer.writerow(dict(stage='dynamics',update=update+1,model_loss=loss))
        for update in range(config.get('actor_pretrain_updates',0)):
            loss, metrics = update_networks(actor,model,transform,actor_opt,model_opt,replay,config)
            writer.writerow(dict(stage='joint',update=update+1,model_loss=loss,**metrics))
            if (update+1) % 32 == 0:
                stream.flush()
                print(json.dumps(dict(pretrain_update=update+1,**metrics)),flush=True)


def run(config, output, saved, phase, seeds, make_plots=True):
    source = torch.load(ROOT/config['checkpoint'], map_location='cpu', weights_only=False)
    transform = Transform(source)
    validate_config(config, transform)
    actor, model = Actor(source), Dynamics(source)
    hidden = source['config']['hidden_size']
    actor.load_state_dict(adapt_state(saved['actor'], True, hidden))
    model.load_state_dict(adapt_state(saved['dynamics'], False, hidden))
    training = phase == 'train'
    actor.train(training)
    model.train(training)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=config['actor_lr'])
    model_opt = torch.optim.Adam(model.parameters(), lr=config['model_lr'])
    replay = Replay(config['buffer_size'], config['seed'], config['regularization_window'])
    rng = np.random.default_rng(config['seed'])
    if output == ROOT/'exp_7/runs' and output.exists():
        # Search trials may coexist with a new final run, but never overwrite results.
        if any(not p.is_dir() or not p.name.startswith('tune_') for p in output.iterdir()):
            raise FileExistsError(f'Run output already exists: {output}')
    else:
        output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps({**config, 'phase':phase, 'seeds':list(seeds)}, indent=2)+'\n')
    backend = DockerSimulator(config['docker_image'])
    summaries, updates = [], 0
    try:
        env = Environment(backend, config, transform.window)
        warmup_dir = output/'warmup'
        warmup_dir.mkdir()
        if training:
            seed_replay(env, replay, config, saved, warmup_dir)
            pretrain(actor,model,transform,actor_opt,model_opt,replay,config,output)
            updates = config.get('actor_pretrain_updates',0)
        with (output/'steps.csv').open('w', newline='') as stream:
            fields = ['phase', 'episode', 'seed', 'step', 'cost', 'predicted_cost', 'model_loss', *METRICS,
                      'safe_step', 'safe_target_step', 'reward', 'released', 'release_reason',
                      'terminated', 'truncated', 'cv_violation', 'termination_reason', *COLUMNS]
            fields += [f'candidate_MV[{i}]' for i in range(11)]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for episode, seed in enumerate(seeds):
                levels = env.reset(seed)
                bounds = env.bounds
                (warmup_dir/f'episode_{episode:03d}_bounds.json').write_text(json.dumps(bounds, indent=2)+'\n')
                pd.DataFrame(env.warmup, columns=COLUMNS).to_csv(warmup_dir/f'episode_{episode:03d}.csv', index=False)
                initial_violation = env.cv_violation
                costs = []
                safe_steps = target_steps = streak = longest_streak = 0
                while not env.done:
                    with torch.no_grad():
                        history, last = tensors(levels, transform)
                        action = actor(history, last, transform)[0].numpy()
                    if training:
                        action = np.clip(action+rng.normal(0,config['exploration_std'],11),0,100).astype(np.float32)
                    following, value, terminated, truncated = env.step(np.append(action, bounds['fixed_mv11']))
                    with torch.no_grad():
                        cv = transform.cv_levels(last, model(history,last,torch.tensor(action)[None],transform))
                        predicted = float(cost(cv, transform.frequency)[0])
                    metrics = {key:'' for key in METRICS}
                    model_loss = ''
                    if training:
                        replay.add(levels, action, following[-1], bounds, config['episode_steps']-env.steps+1)
                        for _ in range(config.get('updates_per_step',1)):
                            model_loss, metrics = update_networks(actor,model,transform,actor_opt,model_opt,replay,config)
                            updates += 1
                    safe = not terminated
                    on_target = safe and abs(value-config['target_cost']) <= config['tracking_tolerance']
                    reward = float(safe)*(config['survival_weight'] + config['tracking_weight']*
                                          np.exp(-((value-config['target_cost'])/config['cost_scale'])**2))
                    writer.writerow(dict(phase=phase,episode=episode,seed=seed,step=env.steps,cost=value,
                        predicted_cost=predicted,model_loss=model_loss,**metrics,safe_step=safe,
                        safe_target_step=on_target,reward=reward,released=False,release_reason='',
                        terminated=terminated,truncated=truncated,cv_violation=env.cv_violation,
                        termination_reason=env.termination_reason,**observation_fields(following),
                        **{f'candidate_MV[{i}]':float(action[i]) for i in range(11)}))
                    costs.append(value)
                    safe_steps += int(safe)
                    target_steps += int(on_target)
                    streak = streak+1 if on_target else 0
                    longest_streak = max(longest_streak, streak)
                    levels = following
                summary = summarize_episode(phase,seed,costs,safe_steps,target_steps,longest_streak,env,initial_violation)
                summaries.append(summary)
                stream.flush()
                print(json.dumps(summary, allow_nan=False), flush=True)
    finally:
        backend.close()
    result = aggregate(summaries, updates, config)
    offline_actor = config.get('actor_pretrain_updates',0) if training else 0
    offline_model = config.get('model_pretrain_updates',0) if training else 0
    result.update(pretrain_actor_updates=offline_actor,pretrain_model_updates=offline_model+offline_actor,
                  online_actor_updates=updates-offline_actor,model_updates=updates+offline_model)
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if training:
        torch.save(dict(actor=actor.state_dict(),dynamics=model.state_dict(),config=config,
                        source_config=source['config'],normalization=source['normalization'],
                        active_mv=list(range(11)),parent_controller=config['parent_controller'],
                        actor_updates=updates,total_steps=result['total_steps']),output/'controller.pt')
    if make_plots:
        for episode in range(len(summaries)):
            plot_run(output, episode, phase)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='.', help='Name under exp_7/runs; default: current run directly in runs')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--phase', choices=('train','validation','evaluation'), default='train')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot rendering during parameter search')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.run != '.' and (Path(args.run).name != args.run or args.run == '..'):
        parser.error('Invalid run name')
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    output = ROOT/'exp_7/runs'/args.run
    if args.phase == 'train':
        config = json.loads(args.config.read_text())
        saved = torch.load(ROOT/config['parent_controller'], map_location='cpu', weights_only=False)
        seeds = range(config['train_seed_start'], config['train_seed_start']+config['episodes'])
    else:
        saved = torch.load(output/'controller.pt', map_location='cpu', weights_only=False)
        config = saved['config']
        default_seeds = list(range(700,705)) if args.phase == 'validation' else list(range(800,810))
        seeds = config.get(f'{args.phase}_seeds',default_seeds)
        output = output/args.phase
    torch.manual_seed(config['seed'])
    run(config, output, saved, args.phase, seeds,make_plots=not args.no_plots)


if __name__ == '__main__':
    main()
