from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT.parent))

from experiment_runner import launch

if __name__ == "__main__":
    launch(EXPERIMENT)
