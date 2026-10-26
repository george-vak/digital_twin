import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class NativeSimulator:
    def __init__(self):
        from temain import TemainProcess, TemainAnalyser
        self.process_type = TemainProcess
        self.analyser_type = TemainAnalyser
        self.process = None

    def reset(self, seed, warmup_steps, window, frequency_seconds):
        self.process = self.process_type(freq=f"{frequency_seconds}s", seed=seed)
        self.analyser = self.analyser_type(freq=f"{frequency_seconds}s")

        count = max(warmup_steps, window + 1)
        tail = self.process.simulate(count).to_numpy()[-(window + 1):]
        if not np.isfinite(tail).all():
            raise RuntimeError("Non-finite simulator warmup")
        if np.all(np.abs(np.diff(tail[-4:, :41], axis=0)) <= 1e-10):
            raise RuntimeError("Simulator is frozen after warmup")
        return tail

    def step(self, action=None):
        command = None if action is None else np.asarray(action, dtype=np.float64)[None, :]
        row = self.process.simulate(1, MV=command).to_numpy()[0]
        return row, float(self.analyser(row[None, :])[0])

    def close(self):
        self.process = None


class DockerSimulator:
    def __init__(self, image):
        self.worker = subprocess.Popen([
            "docker", "run", "--rm", "-i", "--platform", "linux/amd64",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{ROOT}:/app:ro", "-w", "/app",
            image, "python", "-u", "exp_3/simulator.py", "--serve",
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def request(self, **message):
        self.worker.stdin.write(json.dumps(message) + "\n")
        self.worker.stdin.flush()
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError("Simulator container exited; check Docker stderr above")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def reset(self, **kwargs):
        return np.asarray(self.request(op="reset", **kwargs), dtype=np.float64)

    def step(self, action=None):
        row, value = self.request(op="step", action=None if action is None else np.asarray(action).tolist())
        return np.asarray(row, dtype=np.float64), value

    def close(self):
        if self.worker.poll() is None:
            self.worker.stdin.close()
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker.terminate()
                self.worker.wait(timeout=10)
        self.worker.stdout.close()


class Environment:
    def __init__(self, backend, config, window):
        self.backend, self.config, self.window = backend, config, window

    def reset(self, seed):
        self.levels = self.backend.reset(
            seed=seed, warmup_steps=self.config["warmup_steps"],
            window=self.window, frequency_seconds=self.config["frequency_seconds"])
        self.frozen = 0
        self.steps = 0
        return self.levels.copy()

    def step(self, action=None):
        row, value = self.backend.step(action)
        if not np.isfinite(row).all() or not np.isfinite(value):
            raise FloatingPointError("Non-finite simulator observation/cost")
        unchanged = np.all(np.abs(row[:41] - self.levels[-1, :41]) <= self.config["freeze_atol"])
        self.frozen = self.frozen + 1 if unchanged else 0
        self.levels = np.concatenate((self.levels[1:], row[None, :]))
        self.steps += 1
        terminated = self.frozen >= self.config["freeze_steps"]
        truncated = self.steps >= self.config["episode_steps"] and not terminated
        return self.levels.copy(), value, terminated, truncated


def serve():
    sys.path.insert(0, str(ROOT))

    import os
    output = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    backend = NativeSimulator()
    try:
        for line in sys.stdin:
            try:
                message = json.loads(line)
                operation = message.pop("op")
                if operation == "reset":
                    result = backend.reset(**message).tolist()
                elif operation == "step":
                    row, value = backend.step(**message)
                    result = [row.tolist(), value]
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                response = {"result": result}
            except Exception as error:
                response = {"error": f"{type(error).__name__}: {error}"}
            output.write(json.dumps(response, allow_nan=False) + "\n")
    finally:
        backend.close()


if __name__ == "__main__":
    serve()
