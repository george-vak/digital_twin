import argparse
from collections import deque
import csv
import json
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp_3.networks import Actor, Dynamics, Transform, cost
from exp_4.networks import actor_update, validate_regularization
from exp_3.simulator import DockerSimulator, Environment, NativeSimulator


class Replay:
    def __init__(self, capacity, seed, regularization_window):
        self.regularization_window = regularization_window
        self.rows = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, levels, action, following):
        self.rows.append((levels.astype(np.float32).copy(), np.asarray(action, dtype=np.float32).copy(),
                          following.astype(np.float32).copy()))

    def sample(self, count, transform):
        rows = self.rng.sample(list(self.rows), count)
        levels, actions, following = (torch.tensor(np.stack(values)) for values in zip(*rows))
        last = levels[:, -1]
        reference = levels[:, -self.regularization_window:, 41:].mean(dim=1)
        return transform.history(levels), last, actions, transform.cv_target(last, following), reference


def model_update(model, transform, optimizer, batch):
    history, last, actions, target = batch[:4]
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(history, last, actions, transform), target)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite dynamics loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def tensors(levels, transform):
    values = torch.tensor(levels, dtype=torch.float32)[None]
    return transform.history(values), values[:, -1]


def episode_summary(mode, seed, costs, terminated):
    return dict(mode=mode, seed=seed, steps=len(costs), terminated=terminated,
                mean_cost=float(np.mean(costs)), std_cost=float(np.std(costs)),
                total_cost=float(np.sum(costs)))


def observation_fields(levels):
    columns = [f"CV[{i}]" for i in range(41)] + [f"MV[{i}]" for i in range(12)]
    return dict(zip(columns, levels[-1].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="New directory name under exp_4/runs")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--backend", choices=("auto", "native", "docker"), default="auto")
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("--run must be a directory name")
    config = json.loads(args.config.read_text())
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    checkpoint_path = ROOT / config["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    transform = Transform(checkpoint)
    validate_regularization(config, transform.window + 1)
    if config["frequency_seconds"] != transform.frequency:
        raise ValueError("Simulator frequency must match the pretrained model")
    actor, model = Actor(checkpoint), Dynamics(checkpoint)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["actor_lr"])
    model_optimizer = torch.optim.Adam(model.parameters(), lr=config["model_lr"])
    backend_name = args.backend if args.backend != "auto" else ("native" if platform.system() == "Linux" else "docker")
    output = ROOT / "exp_4" / "runs" / args.run
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    backend = NativeSimulator() if backend_name == "native" else DockerSimulator(config["docker_image"])
    try:
        env = Environment(backend, config, transform.window)
        replay = Replay(config["buffer_size"], config["seed"], config["regularization_window"])
        total_steps, actor_updates = 0, 0
        ready = False
        training = []
        with (output / "steps.csv").open("w", newline="") as stream:
            fields = ["phase", "episode", "step", "cost", "predicted_cost", "model_loss", "actor_loss", "actor_cost", "actor_penalty", "actor_regularization", "terminated", "truncated"]
            fields += [f"CV[{i}]" for i in range(41)] + [f"MV[{i}]" for i in range(12)]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for episode in range(config["episodes"]):
                seed = config["train_seed_start"] + episode
                levels = env.reset(seed)
                costs = []
                while True:
                    with torch.no_grad():
                        history, last = tensors(levels, transform)
                        action = actor(history, last, transform)[0].numpy()
                    action = np.clip(action + rng.normal(0, config["exploration_std"], 12), 0, 100).astype(np.float32)
                    following, value, terminated, truncated = env.step(action)

                    with torch.no_grad():
                        prediction = transform.cv_levels(last, model(history, last, torch.tensor(action)[None], transform))
                        predicted_cost = float(cost(prediction, transform.frequency)[0])
                    replay.add(levels, action, following[-1])
                    total_steps += 1
                    model_loss = ""
                    actor_metrics = dict(actor_loss="", actor_cost="", actor_penalty="", actor_regularization="")
                    if total_steps >= config["initial_collect_steps"]:
                        if not ready:
                            for _ in range(config["model_pretrain_updates"]):
                                model_update(model, transform, model_optimizer, replay.sample(config["batch_size"], transform))
                            ready = True
                        batch = replay.sample(config["batch_size"], transform)
                        model_loss = model_update(model, transform, model_optimizer, batch)
                        actor_metrics = actor_update(
                            actor, model, transform, actor_optimizer, batch[0], batch[1], batch[4],
                            config["regularization_weight"], config["regularization_scale"])
                        actor_updates += 1
                    writer.writerow(dict(phase="train", episode=episode, step=env.steps, cost=value,
                                         predicted_cost=predicted_cost, model_loss=model_loss, **actor_metrics,
                                         terminated=terminated, truncated=truncated,
                                         **observation_fields(following)))
                    costs.append(value)
                    levels = following
                    if terminated or truncated:
                        break
                stream.flush()
                summary = episode_summary("train", seed, costs, terminated)
                training.append(summary)
                print(json.dumps(summary), flush=True)
            torch.save(dict(actor=actor.state_dict(), dynamics=model.state_dict(),
                            source_config=checkpoint["config"], normalization=checkpoint["normalization"],
                            config=config, total_steps=total_steps, actor_updates=actor_updates), output / "controller.pt")
        (output / "summary.json").write_text(json.dumps(dict(training=training,
                         total_steps=total_steps, actor_updates=actor_updates), indent=2) + "\n")
        from exp_4.plot import plot_run
        plot_run(output)
    finally:
        backend.close()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
