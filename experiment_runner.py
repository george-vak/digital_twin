import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from experiment_paths import ROOT, relative_path, resolve_path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch(experiment):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"),
                        help="New run name; existing runs are never overwritten")
    parser.add_argument("--train", action="store_true", help="Train fresh models before evaluation")
    parser.add_argument("--source-run", default="original", help="Use saved models from this run without --train")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing or executing")
    parser.add_argument("--overwrite", action="store_true", help="Retrain exp_2 in an existing run, replacing its results")
    parser.add_argument("--keep-report", action="store_true", help="Do not rewrite the text report")
    args = parser.parse_args()
    for name in (args.run, args.source_run):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            parser.error("Run names may contain letters, digits, _ and -")
    config_path = experiment / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["protocol"] not in ("levels_standardization_mae", "differences_standardization_mase"):
        parser.error("Unknown experiment protocol")
    differences = config["protocol"] == "differences_standardization_mase"
    run = experiment / "runs" / args.run
    if args.overwrite and (not differences or not args.train or args.run == "original"):
        parser.error("--overwrite requires --train for a non-original exp_2 run")
    if run.exists() and not args.overwrite:
        parser.error(f"Run already exists: {run}. Choose a new --run")
    data = resolve_path(config["training_data"])
    tests = [resolve_path(p) for p in config["test_files"]]
    data_files = [data / f"{split}.csv.gz" for split in ("train", "validation")] + tests
    inputs = set(data_files)
    for file in data_files:
        inputs.update(file.parent / name for name in ("metadata.json", "normalization.json", "variables.csv"))
    if differences:
        if config["horizons"] != [1] or tests != [data / "test.csv.gz"]:
            parser.error("Experiment 2 compares h=1 on the original dataset's test split")
        inputs.add(resolve_path(config["reference_model"]))
    for path in inputs:
        if not path.is_file():
            parser.error(f"Input not found: {path}")
    for folder in {p.parent for p in data_files}:
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["frequency_seconds"] != config["frequency_seconds"]:
            parser.error(f"Frequency does not match config: {folder}")

    commands, copies = [], []
    training_data = run / "data/differenced" if differences else data
    reuse_data = args.overwrite and training_data.is_dir()
    if reuse_data:
        previous = json.loads((run / "manifest.json").read_text())
        for path in inputs:
            if previous["input_sha256"].get(relative_path(path)) != sha256(path):
                parser.error(f"Cannot reuse prepared data: input changed: {path}")
    if differences and args.train and not reuse_data:
        commands.append([sys.executable, str(ROOT / "preprocess.py"), "--data", str(data), "--output", str(training_data)])
    for horizon in config["horizons"]:
        name = "gru.pt" if horizon == 1 else f"gru_h{horizon}.pt"
        model = run / "models" / name
        if args.train:
            command = [sys.executable, str(ROOT / "model.py"), "--data", str(training_data),
                       "--run-dir", str(run), "--horizon", str(horizon)]
            for key, value in config["training"].items():
                command += ["--" + key.replace("_", "-"), str(value)]
            if args.overwrite:
                command.append("--overwrite")
            commands.append(command)
        else:
            source = experiment / "runs" / args.source_run / "models" / name
            if not source.is_file():
                parser.error(f"Model not found: {source}; use --train or another --source-run")
            copies.append((source, model))
            history = source.with_suffix(".history.csv")
            if history.exists():
                copies.append((history, model.with_suffix(".history.csv")))
        if differences:
            commands.append([sys.executable, str(ROOT / "evaluate_mase.py"), "--data", str(data),
                             "--reference-model", str(resolve_path(config["reference_model"])),
                             "--model", str(model), "--output-dir", str(run)])
            if args.keep_report:
                commands[-1].append("--skip-report")
            continue
        for test in tests:
            stem = test.name.removesuffix(".gz").removesuffix(".csv")
            report = run / "results" / f"{model.stem}_{test.parent.name}_{stem}.csv"
            commands.append([sys.executable, str(ROOT / "eval.py"), str(test), "--model", str(model)])
            commands.append([sys.executable, str(ROOT / "compare.py"), str(report)])
    if not differences:
        commands.append([sys.executable, str(ROOT / "grf.py"), "--run-dir", str(run), "--no-show"])
    if args.dry_run:
        for source, target in copies:
            print(f"Copy {relative_path(source)} -> {relative_path(target)}")
        for command in commands:
            print(shlex.join(command))
        return

    run.mkdir(parents=True, exist_ok=args.overwrite)
    (run / "models").mkdir(exist_ok=args.overwrite)
    (run / "logs").mkdir(exist_ok=args.overwrite)
    shutil.copy2(config_path, run / "config.json")
    code_files = sorted(ROOT.glob("*.py")) + [experiment / "run.py", ROOT / "requirements.txt", ROOT / "Dockerfile"]
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment.name, "run": args.run, "trained": args.train,
        "source_run": None if args.train else args.source_run,
        "config": config, "python": sys.version,
        "packages": {name: version(name) for name in ("numpy", "pandas", "torch", "matplotlib")},
        "code_sha256": {relative_path(p): sha256(p) for p in code_files},
        "input_sha256": {relative_path(p): sha256(p) for p in sorted(inputs)},
        "copied_sha256": {relative_path(target): sha256(source) for source, target in copies},
        "commands": commands, "completed_commands": 0,
        "overwrite": args.overwrite, "report_preserved": args.keep_report,
    }
    manifest_path = run / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    try:
        for source, target in copies:
            shutil.copy2(source, target)
        with (run / "logs/run.log").open("w", encoding="utf-8") as log:
            for command in commands:
                print(shlex.join(command), flush=True)
                log.write("$ " + shlex.join(command) + "\n")
                log.flush()
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    if process.wait():
                        raise RuntimeError(f"Command failed ({process.returncode}): {shlex.join(command)}")
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
                    process.stdout.close()
                manifest["completed_commands"] += 1
                save_manifest()
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error) or type(error).__name__
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
    print(f"Run saved: {relative_path(run)}")
