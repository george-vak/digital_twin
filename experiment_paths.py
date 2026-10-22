import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = ROOT / "exp_1"
DEFAULT_RUN = DEFAULT_EXPERIMENT / "runs/original"
DEFAULT_DATA = DEFAULT_EXPERIMENT / "data/base_20_5"


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def relative_path(path):
    return Path(os.path.relpath(resolve_path(path).resolve(), ROOT)).as_posix()


def run_for_artifact(path):
    for parent in resolve_path(path).resolve().parents:
        if parent.parent.name == "runs":
            return parent
    return DEFAULT_RUN


def training_data_path(path, model_path):
    """Resolve the old dataset label without rewriting archived checkpoints."""
    resolved = resolve_path(path)
    if resolved.exists():
        return resolved
    # Original .pt files retain the absolute path of the historical checkout.
    parts = Path(path).parts
    if "data" in parts:
        suffix = Path(*parts[parts.index("data") + 1:])
        relocated = run_for_artifact(model_path).parent.parent / "data" / suffix
        if relocated.exists():
            return relocated
    return resolved
