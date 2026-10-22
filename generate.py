import argparse
import json
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_paths import DEFAULT_EXPERIMENT, ROOT

DOCKER_IMAGE = "temain-simulator:local"
SPLITS = ("train", "validation", "test")
DEFAULT_SEEDS = {"train": (1, 2, 3, 4), "validation": (6,), "test": (7,)}
CLI_SPLITS = {"train": "tr", "validation": "val", "test": "test"}


def run_in_docker(args) -> None:
    """Run the Linux x86-64 simulator with the project mounted at /app."""
    if shutil.which("docker") is None:
        raise SystemExit("Docker not found. Install and start Docker Desktop, then run this command again.")
    daemon = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if daemon.returncode:
        raise SystemExit(f"Docker is unavailable. Start Docker Desktop and try again.\n{daemon.stderr.strip()}")
    image = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", DOCKER_IMAGE],
        capture_output=True, text=True,
    )
    if image.returncode or image.stdout.strip() != "linux/amd64":
        print("Building the simulator image (first run)...", flush=True)
        result = subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", DOCKER_IMAGE, str(ROOT)])
        if result.returncode:
            raise SystemExit(result.returncode)

    command = ["docker", "run", "--rm", "--platform", "linux/amd64",
               "-v", f"{ROOT}:/app", "-w", "/app"]
    if args.normalization:
        source = args.normalization.resolve()
        if not source.is_file():
            raise SystemExit(f"Normalization file not found: {source}")
        command += ["-v", f"{source}:/normalization.json:ro"]
    command += [DOCKER_IMAGE, "python", "generate.py"]
    for name, value in vars(args).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif value is not None:
            if name == "normalization":
                value = "/normalization.json"
            elif name == "experiment":
                value = Path("/app") / value.resolve().relative_to(ROOT)
            elif isinstance(value, tuple):
                value = ",".join(map(str, value))
            command += [flag, str(value)]
    print("Generating data in Docker...", flush=True)
    result = subprocess.run(command)
    if result.returncode == 0:
        print(f"Local dataset: {args.experiment / 'data' / args.dataset}", flush=True)
    raise SystemExit(result.returncode)


def fit_normalization(train: pd.DataFrame, columns: list[str]) -> dict:
    """Fit per-variable z-score parameters on the complete training split."""
    values = train[columns]
    mean = values.mean()
    std = values.std(ddof=0)
    constant = std.eq(0)
    return {
        "method": "standardization",
        "fit_split": "train",
        "columns": columns,
        "mean": mean.to_dict(),
        "scale": std.mask(constant, 1.0).to_dict(),
        "constant_columns": std.index[constant].tolist(),
        "ddof": 0,
        "transform": "(x - mean) / scale",
        "inverse_transform": "x_normalized * scale + mean",
    }


def normalize_frame(frame: pd.DataFrame, normalization: dict) -> pd.DataFrame:
    """Apply saved train parameters, preserving run and time columns."""
    result = frame.copy()
    columns = normalization["columns"]
    result[columns] = (result[columns] - pd.Series(normalization["mean"])) / pd.Series(normalization["scale"])
    if not np.isfinite(result[columns].to_numpy()).all():
        raise RuntimeError("Normalization produced non-finite values")
    return result


def preprocessing_metadata() -> dict:
    return {
        "normalization": "Per-variable standardization fitted on train only",
        "parameters_file": "normalization.json",
        "unscaled_columns": ["run_id", "step", "elapsed_seconds"],
        "windows": "Not created. Create within each run_id and split for the chosen model.",
    }


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc
    if not seeds or any(seed == 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be distinct nonzero integers")
    return seeds


def parse_duration(value: str) -> pd.Timedelta:
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Use a duration such as 5000min or 48h") from exc
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise argparse.ArgumentTypeError("Duration must be positive")
    return duration


def generate_run(seed: int, steps: int, frequency: str, frequency_seconds: int,
                 chunk_steps: int) -> pd.DataFrame:
    from temain import NCV, NMV, TemainProcess

    process = TemainProcess(freq=frequency, seed=seed)
    chunks = []
    for start in range(0, steps, chunk_steps):
        count = min(chunk_steps, steps - start)
        values = process.simulate(count).reset_index(drop=True)
        if values.shape != (count, NCV + NMV) or not np.isfinite(values.to_numpy()).all():
            raise RuntimeError(f"Seed {seed}: unexpected shape or non-finite values")
        values.insert(0, "elapsed_seconds", np.arange(start, start + count, dtype=np.int64) * frequency_seconds)
        values.insert(0, "step", np.arange(start, start + count, dtype=np.int64))
        values.insert(0, "run_id", f"seed_{seed}")
        chunks.append(values)
    print(f"seed {seed}: {steps} samples at {frequency}", flush=True)
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--h", "-h", "--help", action="help", help="Show help and exit")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT, help="Experiment directory inside the repository")
    parser.add_argument("--dataset", default="base_20_5", help="Name of a new directory inside exp_i/data/")
    parser.add_argument("--freq", "--frequency", dest="frequency", default="60s", help="Sampling interval with units, e.g. 10s, 60s, 5min")
    parser.add_argument("--tr-dur", "--train-duration", dest="train_duration", type=parse_duration, default=parse_duration("20000min"))
    parser.add_argument("--val-dur", "--validation-duration", dest="validation_duration", type=parse_duration, default=parse_duration("5000min"))
    parser.add_argument("--test-dur", "--test-duration", dest="test_duration", type=parse_duration, default=parse_duration("5000min"))
    parser.add_argument("--tonly", "--test-only", dest="test_only", action="store_true", help="Generate only test, using an existing normalization")
    parser.add_argument("--nrm", "--normalization", dest="normalization", type=Path, help="Existing normalization.json; required for --tonly")
    for split in SPLITS:
        flags = list(dict.fromkeys((f"--{CLI_SPLITS[split]}-seeds", f"--{split}-seeds")))
        parser.add_argument(*flags, dest=f"{split}_seeds", type=parse_seeds,
                            default=DEFAULT_SEEDS[split], help="Comma-separated simulator seeds")
    parser.add_argument("--chunk", "--chunk-steps", dest="chunk_steps", type=int, default=1000,
                        help="Samples per simulator call; controls peak memory")
    args = parser.parse_args()
    args.experiment = args.experiment.resolve()
    if not args.experiment.is_relative_to(ROOT):
        parser.error("--experiment must be inside the repository (also mounted in Docker)")
    data_root = args.experiment / "data"

    if args.test_only and args.normalization is None:
        parser.error("--tonly requires --nrm from the training dataset")
    splits = ("test",) if args.test_only else SPLITS

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.dataset):
        parser.error("--dataset must be a simple folder name: letters, digits, _ or -")
    if args.chunk_steps < 1:
        parser.error("--chunk must be positive")
    try:
        frequency = pd.Timedelta(args.frequency)
    except (TypeError, ValueError) as exc:
        parser.error(f"Invalid --freq: {exc}")
    if pd.isna(frequency) or frequency < pd.Timedelta("1s") or frequency.value % pd.Timedelta("1s").value:
        parser.error("--freq must be a positive whole number of seconds with units, e.g. 60s (not 60)")
    frequency_seconds = int(frequency.total_seconds())

    seeds = {split: getattr(args, f"{split}_seeds") for split in splits}
    all_seeds = [seed for group in seeds.values() for seed in group]
    if len(all_seeds) != len(set(all_seeds)):
        parser.error("A seed may appear in only one split")

    settings = {}
    for split in splits:
        duration = getattr(args, f"{split}_duration")
        divisor = frequency.value * len(seeds[split])
        if duration.value % divisor:
            parser.error(f"--{CLI_SPLITS[split]}-dur must be divisible by frequency × number of seeds")
        steps_per_run = duration.value // divisor
        if steps_per_run < 2:
            parser.error(f"{split}: each run must have at least two samples")
        settings[split] = {"duration": str(duration), "seeds": list(seeds[split]),
                           "steps_per_run": int(steps_per_run),
                           "rows": int(steps_per_run * len(seeds[split]))}

    output = data_root / args.dataset
    if output.exists():
        parser.error(f"Dataset already exists: {output}. Choose another --dataset name")

    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        run_in_docker(args)

    from temain import NCV, NMV, TemainProcess

    data_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.dataset}-", dir=data_root))
    try:
        normalization = json.loads(args.normalization.read_text()) if args.normalization else None
        columns = TemainProcess.info().index.tolist()
        for split in splits:
            frames = [generate_run(seed, settings[split]["steps_per_run"], args.frequency,
                                   frequency_seconds, args.chunk_steps)
                      for seed in seeds[split]]
            frame = pd.concat(frames, ignore_index=True)
            if split == "train" and normalization is None:
                normalization = fit_normalization(frame, columns)
            normalize_frame(frame, normalization).to_csv(
                staging / f"{split}.csv.gz", index=False, float_format="%.17g")

        (staging / "normalization.json").write_text(
            json.dumps(normalization, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        TemainProcess.info().to_csv(staging / "variables.csv", index_label="variable")
        metadata = {
            "schema_version": 3,
            "source": "temain.py / libtemain.so",
            "scenario": "normal closed-loop operation; no external MV or DV supplied",
            "frequency": args.frequency,
            "frequency_seconds": frequency_seconds,
            "time_columns": {"step": "zero-based sample number within a run",
                             "elapsed_seconds": "time since the run started"},
            "process_columns": {"measurements": NCV, "controls": NMV},
            "splits": settings,
            "preprocessing": preprocessing_metadata(),
        }
        if args.normalization:
            metadata["preprocessing"]["parameters_source"] = str(args.normalization)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Dataset ready: {output}", flush=True)


if __name__ == "__main__":
    main()
