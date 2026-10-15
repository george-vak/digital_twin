import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data/base_20_5"
DEFAULT_MODEL = ROOT / "models/gru.pt"
WINDOW = 60  # Number of input points: t-59, ..., t. Thus L = 59.
HORIZON = 1  # Predict t+h; 1 step = 1 minute for the current dataset.


def default_model_path(horizon):
    return DEFAULT_MODEL if horizon == 1 else ROOT / "models" / f"gru_h{horizon}.pt"


def load_frame(path: Path, normalization: dict, frequency_seconds: int) -> pd.DataFrame:
    """Read generated data and put it on the model's training scale.

    For normalized CSVs, metadata.json and normalization.json must accompany
    the file. Without metadata, the CSV is treated as raw physical values.
    """
    frame = pd.read_csv(path)
    columns = normalization["columns"]
    metadata_path = path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["frequency_seconds"] != frequency_seconds:
            raise ValueError("Data frequency differs from the model's training frequency")
        preprocessing = metadata.get("preprocessing", {})
        if isinstance(preprocessing, dict) and "parameters_file" in preprocessing:
            source = json.loads((path.parent / preprocessing["parameters_file"]).read_text())
            frame[columns] = frame[columns] * pd.Series(source["scale"]) + pd.Series(source["mean"])
    frame[columns] = (frame[columns] - pd.Series(normalization["mean"])) / pd.Series(normalization["scale"])
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError("Process columns contain NaN or infinite values")
    return frame


class WindowDataset(Dataset):
    """Sliding windows with stride 1; windows never cross run boundaries."""

    def __init__(self, frame, columns, window, horizon, frequency_seconds):
        if window < 1 or horizon < 1:
            raise ValueError("Window and horizon must be positive")
        self.window = window
        self.horizon = horizon
        self.runs = []
        self.indices = []
        for run_id, run in frame.groupby("run_id", sort=False):
            run = run.sort_values("step")
            if not ((np.diff(run["step"]) == 1).all()
                    and (np.diff(run["elapsed_seconds"]) == frequency_seconds).all()):
                raise ValueError(f"Run {run_id} has duplicate steps or gaps in time")
            count = len(run) - window - horizon + 1
            if count < 1:
                raise ValueError(f"Run {run_id} is too short for this window and horizon")
            run_index = len(self.runs)
            self.runs.append(torch.from_numpy(run[columns].to_numpy(dtype=np.float32, copy=True)))
            self.indices.extend((run_index, start) for start in range(count))
        if not self.indices:
            raise ValueError("No windows in the input data")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        run_index, start = self.indices[index]
        run = self.runs[run_index]
        end = start + self.window
        return run[start:end], run[end - 1 + self.horizon]


class GRUModel(nn.Module):
    def __init__(self, input_size=53, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


@torch.no_grad()
def validation_mse(model, loader):
    model.eval()
    total = 0.0
    for x, y in loader:
        total += nn.functional.mse_loss(model(x), y, reduction="sum").item()
    return total / (len(loader.dataset) * model.head.out_features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, help="Override models/gru.pt (h=1) or models/gru_hN.pt")
    parser.add_argument("--window", type=int, default=WINDOW, help="Input points, equal to L+1")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="h in t+h, in sampling steps")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.window, args.horizon, args.hidden_size, args.epochs, args.patience, args.batch_size) < 1 or args.lr <= 0:
        parser.error("Sizes, horizon, epochs, patience and learning rate must be positive")
    if args.output is None:
        args.output = default_model_path(args.horizon)

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    normalization = json.loads((args.data / "normalization.json").read_text())
    metadata = json.loads((args.data / "metadata.json").read_text())
    columns = normalization["columns"]
    frequency = metadata["frequency_seconds"]
    loaders = {}
    rows = {}
    for split in ("train", "validation"):
        frame = load_frame(args.data / f"{split}.csv.gz", normalization, frequency)
        dataset = WindowDataset(frame, columns, args.window, args.horizon, frequency)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"))
        rows[split] = len(frame)
        print(f"{split}: {len(frame)} rows, {len(dataset)} windows", flush=True)

    model = GRUModel(len(columns), args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = dict(input_size=len(columns), hidden_size=args.hidden_size,
                  window=args.window, horizon=args.horizon, frequency_seconds=frequency)
    variables = pd.read_csv(args.data / "variables.csv", index_col="variable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(x)
        train_loss = total / len(loaders["train"].dataset)
        val_loss = validation_mse(model, loaders["validation"])
        history.append(dict(epoch=epoch, train_mse=train_loss, validation_mse=val_loss))
        print(f"Epoch {epoch:02d}: train MSE={train_loss:.6f}, validation MSE={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            torch.save({
                "state_dict": model.state_dict(), "config": config,
                "normalization": normalization,
                "variables": variables[["description", "unit"]].to_dict(orient="index"),
                "training": {"dataset": str(args.data.resolve()), "rows": rows,
                             "windows": {s: len(loader.dataset) for s, loader in loaders.items()},
                             "splits": {s: metadata["splits"][s] for s in loaders},
                             "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
                             "best_epoch": epoch, "validation_mse": val_loss},
            }, args.output)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs", flush=True)
                break
    pd.DataFrame(history).to_csv(args.output.with_suffix(".history.csv"), index=False)
    print(f"Best model: {args.output}; validation MSE={best_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
