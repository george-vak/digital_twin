"""Fine tune exp_4 with barrier regularization and release to internal control."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from exp_6.networks import Actor,Dynamics,Transform,cost,adapt_state
from exp_3.simulator import DockerSimulator
from exp_6.data import Environment,Replay
from exp_4.run import model_update,tensors,observation_fields,episode_summary
from exp_4.networks import validate_regularization
from exp_6.networks import actor_update
from exp_6.safety import ReleaseGate
from exp_6.plot import plot_run


def run(config, output, saved, phase, seeds):
    if phase=='train':
        if config.get('actor_objective')!='per_transition_squared_cost_error':
            raise ValueError('Training requires actor_objective=per_transition_squared_cost_error')
        if not isinstance(config.get('target_cost'),(int,float)) or not np.isfinite(config['target_cost']):
            raise ValueError('Training requires finite target_cost')
    source=torch.load(ROOT/config['checkpoint'],map_location='cpu',weights_only=False)
    transform=Transform(source)
    validate_regularization(config,transform.window+1)
    if not 0 < config['barrier_epsilon'] < 1:
        raise ValueError('barrier_epsilon must be between 0 and 1')
    if not np.isfinite(config['barrier_weight']) or config['barrier_weight'] < 0:
        raise ValueError('barrier_weight must be finite and nonnegative')
    actor,model=Actor(source),Dynamics(source)
    hidden=source['config']['hidden_size']
    actor.load_state_dict(adapt_state(saved['actor'],True,hidden))
    model.load_state_dict(adapt_state(saved['dynamics'],False,hidden))
    training=phase=='train'
    actor.train(training); model.train(training)
    opt=torch.optim.Adam(actor.parameters(),lr=config['actor_lr'])
    model_opt=torch.optim.Adam(model.parameters(),lr=config['model_lr'])
    replay=Replay(config['buffer_size'],config['seed'],config['regularization_window'])
    rng=np.random.default_rng(config['seed'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps({**config,'phase':phase,'seeds':list(seeds)},indent=2)+'\n')
    backend=DockerSimulator(config['docker_image'])
    summaries=[]; updates=0; total=0
    metrics_names=['actor_loss','actor_cost','actor_tracking_loss','actor_penalty','actor_regularization','cv_barrier','mv_barrier','actor_barrier']
    try:
        env=Environment(backend,config,transform.window)
        warmup_dir=output/'warmup';warmup_dir.mkdir()
        columns=[f'CV[{i}]' for i in range(41)]+[f'MV[{i}]' for i in range(12)]
        if training:
            frame=pd.read_csv(ROOT/config['replay_source'])
            parent_config=saved['config']
            for episode,group in frame.groupby('episode',sort=False):
                seed=parent_config['train_seed_start']+int(episode)
                env.reset(seed)
                (warmup_dir/f'parent_episode_{int(episode):03d}_bounds.json').write_text(json.dumps(env.bounds,indent=2)+'\n')
                values=group[columns].to_numpy(dtype=np.float32)
                for index in range(transform.window,len(values)-1):
                    replay.add(values[index-transform.window:index+1],values[index+1,41:52],values[index+1],env.bounds)
            if len(replay.rows)<config['batch_size']:raise ValueError('Insufficient parent replay')
        with (output/'steps.csv').open('w',newline='') as stream:
            fields=['phase','episode','seed','step','cost','predicted_cost','model_loss',*metrics_names,
                    'released','release_reason','terminated','truncated']
            fields += [f'CV[{i}]' for i in range(41)]+[f'MV[{i}]' for i in range(12)]
            fields += [f'candidate_MV[{i}]' for i in range(11)]
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for episode,seed in enumerate(seeds):
                levels=env.reset(seed)
                bounds=env.bounds
                (warmup_dir/f'episode_{episode:03d}_bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
                pd.DataFrame(env.warmup,columns=columns).to_csv(warmup_dir/f'episode_{episode:03d}.csv',index=False)
                gate=ReleaseGate(bounds)
                costs=[]; released_count=0
                while True:
                    with torch.no_grad():
                        history,last=tensors(levels,transform)
                        action=actor(history,last,transform)[0].numpy()
                    if training:
                        action=np.clip(action+rng.normal(0,config['exploration_std'],11),0,100).astype(np.float32)
                    released,reason=gate.decide(levels[-1],action)
                    following,value,terminated,truncated=env.step(None if released else np.append(action,bounds['fixed_mv11']))
                    predicted=''
                    metrics={k:'' for k in metrics_names}; model_loss=''
                    if not released:
                        with torch.no_grad():
                            cv=transform.cv_levels(last,model(history,last,torch.tensor(action)[None],transform))
                            predicted=float(cost(cv,transform.frequency)[0])
                        if training:
                            replay.add(levels,action,following[-1],bounds)
                    if training and len(replay.rows)>=config['batch_size']:
                        batch=replay.sample(config['batch_size'],transform)
                        model_loss=model_update(model,transform,model_opt,batch)
                        metrics=actor_update(actor,model,transform,opt,batch,config)
                        updates+=1
                    # Native fallback varies within a minute; do not label it as a constant actor action.
                    writer.writerow(dict(phase=phase,episode=episode,seed=seed,step=env.steps,cost=value,
                        predicted_cost=predicted,model_loss=model_loss,**metrics,released=released,
                        release_reason=reason,terminated=terminated,truncated=truncated,
                        **observation_fields(following),**{f'candidate_MV[{i}]':float(action[i]) for i in range(11)}))
                    costs.append(value); released_count+=int(released); total+=1; levels=following
                    if terminated or truncated: break
                summary=episode_summary(phase,seed,costs,terminated)
                summary.update(released_steps=released_count,actor_steps=len(costs)-released_count,
                               release_fraction=released_count/len(costs))
                if 'target_cost' in config:
                    errors=np.asarray(costs)-config['target_cost']
                    summary.update(target_cost=config['target_cost'],
                                   cost_mae=float(np.abs(errors).mean()),cost_rmse=float(np.sqrt(np.mean(errors**2))))
                summaries.append(summary); stream.flush(); print(json.dumps(summary),flush=True)
    finally:
        backend.close()
    result=dict(episodes=summaries,total_steps=total,actor_updates=updates,
                mean_steps=float(np.mean([s['steps'] for s in summaries])),
                mean_cost=float(np.mean([s['mean_cost'] for s in summaries])),
                release_fraction=sum(s['released_steps'] for s in summaries)/total)
    if 'target_cost' in config:
        result.update(target_cost=config['target_cost'],
                      cost_mae=sum(s['cost_mae']*s['steps'] for s in summaries)/total,
                      cost_rmse=float(np.sqrt(sum(s['cost_rmse']**2*s['steps'] for s in summaries)/total)))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if training:
        torch.save(dict(actor=actor.state_dict(),dynamics=model.state_dict(),config=config,
            source_config=source['config'],normalization=source['normalization'],active_mv=list(range(11)),
            parent_controller=config['parent_controller'],actor_updates=updates,total_steps=total),output/'controller.pt')
    for episode in range(len(summaries)):
        plot_run(output,episode,phase)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--phase',choices=('train','validation','evaluation'),default='train')
    args=parser.parse_args()
    if Path(args.run).name!=args.run or args.run in ('.','..'): parser.error('Invalid run name')
    torch.set_num_threads(4)
    output=ROOT/'exp_6/runs'/args.run
    if args.phase=='train':
        config=json.loads(args.config.read_text()); torch.manual_seed(config['seed'])
        saved=torch.load(ROOT/config['parent_controller'],map_location='cpu',weights_only=False)
        seeds=range(config['train_seed_start'],config['train_seed_start']+config['episodes'])
    else:
        saved=torch.load(output/'controller.pt',map_location='cpu',weights_only=False)
        config=saved['config']
        seeds=range(700,705) if args.phase=='validation' else range(800,810)
        output=output/args.phase
    run(config,output,saved,args.phase,seeds)


if __name__=='__main__': main()
