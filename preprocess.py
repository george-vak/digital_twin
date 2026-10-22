"""First differences within runs, then train-only centering and scaling."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiment_paths import relative_path
from generate import fit_normalization, normalize_frame
from model import load_physical_frame


def difference_frame(frame, columns, frequency_seconds=60):
    pieces = []
    for run_id, run in frame.groupby("run_id", sort=False):
        run = run.sort_values("step")
        if len(run) < 2 or not ((np.diff(run["step"]) == 1).all()
                               and (np.diff(run["elapsed_seconds"]) == frequency_seconds).all()):
            raise ValueError(f"Run {run_id}: need continuous observations before differencing")
        delta = run.copy()
        delta[columns] = run[columns].diff()
        pieces.append(delta.iloc[1:])
    if not pieces:
        raise ValueError("Empty data")
    return pd.concat(pieces, ignore_index=True)


def prepare(source, output):
    if output.exists():
        raise ValueError(f"Prepared dataset already exists: {output}")
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["frequency_seconds"] != 60 or metadata.get("target_transform", "levels") != "levels":
        raise ValueError("Experiment 2 requires levels sampled every 60 seconds")
    columns = json.loads((source / "normalization.json").read_text())["columns"]
    frames = {split: difference_frame(load_physical_frame(source / f"{split}.csv.gz", columns, 60), columns)
              for split in ("train", "validation", "test")}
    normalization = fit_normalization(frames["train"], columns)
    output.mkdir(parents=True)
    for split, frame in frames.items():
        normalize_frame(frame, normalization).to_csv(output / f"{split}.csv.gz", index=False, float_format="%.17g")
    (output / "normalization.json").write_text(json.dumps(normalization, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(source / "variables.csv", output / "variables.csv")
    metadata["source_dataset"] = relative_path(source.resolve())
    metadata["target_transform"] = "first_difference"
    metadata["preprocessing"] = {
        "normalization": "First difference per run, then (delta - train_mean) / train_std; ddof=0",
        "parameters_file": "normalization.json",
        "unscaled_columns": ["run_id", "step", "elapsed_seconds"],
    }
    for split, frame in frames.items():
        metadata["splits"][split]["source_rows"] = metadata["splits"][split]["rows"]
        metadata["splits"][split]["rows"] = len(frame)
        metadata["splits"][split]["steps_per_run"] -= 1
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared differences: {output}; " + ", ".join(f"{s}={len(f)}" for s, f in frames.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.output)
