"""Evaluate a frozen controller on simulator seeds without exploration noise."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp_3.networks import Actor, Dynamics, Transform, cost
from exp_3.simulator import DockerSimulator, Environment
from exp_4.run import tensors, observation_fields, episode_summary
from exp_4.plot import plot_run


def evaluate(run_dir, seeds, phase="evaluation", episode_steps=200, all_plots=False):
    run_dir = Path(run_dir)
    saved = torch.load(run_dir / "controller.pt", map_location="cpu", weights_only=False)
    config = {**saved["config"], "episode_steps": episode_steps,
              "evaluation_seeds": list(seeds), "exploration_std": 0.0, "phase": phase}
    source = torch.load(ROOT / config["checkpoint"], map_location="cpu", weights_only=False)
    transform = Transform(source)
    actor, model = Actor(source), Dynamics(source)
    actor.load_state_dict(saved["actor"])
    model.load_state_dict(saved["dynamics"])
    actor.eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    output = run_dir / phase
    output.mkdir(exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    backend = DockerSimulator(config["docker_image"])
    summaries = []
    try:
        env = Environment(backend, config, transform.window)
        with (output / "steps.csv").open("w", newline="") as stream, torch.no_grad():
            fields = ["phase", "episode", "seed", "step", "cost", "predicted_cost", "actor_loss", "terminated", "truncated"]
            fields += [f"CV[{i}]" for i in range(41)] + [f"MV[{i}]" for i in range(12)]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for episode, seed in enumerate(seeds):
                levels = env.reset(seed)
                costs = []
                deviations, changes = [], []
                while True:
                    history, last = tensors(levels, transform)
                    action_tensor = actor(history, last, transform)
                    action = action_tensor[0].numpy()
                    reference = levels[-config["regularization_window"]:, 41:].mean(axis=0)
                    deviations.append(float(np.mean((action - reference) ** 2)))
                    changes.append(float(np.mean((action - levels[-1, 41:]) ** 2)))
                    prediction = transform.cv_levels(last, model(history, last, action_tensor, transform))
                    following, value, terminated, truncated = env.step(action)
                    writer.writerow(dict(phase=phase, episode=episode, seed=seed, step=env.steps,
                                         cost=value, predicted_cost=float(cost(prediction, transform.frequency)[0]),
                                         actor_loss="", terminated=terminated, truncated=truncated,
                                         **observation_fields(following)))
                    costs.append(value)
                    levels = following
                    if terminated or truncated:
                        break
                summary = episode_summary(phase, seed, costs, terminated)
                summary.update(mv_window_rmse=float(np.sqrt(np.mean(deviations))),
                               mv_step_rmse=float(np.sqrt(np.mean(changes))))
                summaries.append(summary)
                stream.flush()
                print(json.dumps(summary), flush=True)
    finally:
        backend.close()
    aggregate = dict(episodes=summaries,
                     mean_steps=float(np.mean([s["steps"] for s in summaries])),
                     completed=sum(not s["terminated"] for s in summaries),
                     mean_cost=float(np.mean([s["mean_cost"] for s in summaries])))
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    if all_plots:
        for episode in range(len(summaries)):
            plot_run(output, episode=episode, phase=phase)
    else:
        plot_run(output, phase=phase)
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--phase", choices=("validation", "evaluation"), default="evaluation")
    parser.add_argument("--seed-start", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--all-plots", action="store_true")
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("--run must be a directory name")
    if args.episodes < 1 or args.episode_steps < 1:
        parser.error("Episode count and length must be positive")
    torch.set_num_threads(4)
    evaluate(ROOT / "exp_4/runs" / args.run, range(args.seed_start, args.seed_start + args.episodes),
             args.phase, args.episode_steps, args.all_plots)


if __name__ == "__main__":
    main()
