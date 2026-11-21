"""Expose the internal controller's nominal MV using an independent simulator clone."""
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp_3 import simulator


class NativeSimulator(simulator.NativeSimulator):
    def nominal(self):
        return self.process.clone().simulate(1).to_numpy()[0, 41:].tolist()

    def branches(self, actions, horizon):
        import numpy as np
        results = []
        for action in actions:
            process = self.process.clone()
            if action is None:
                rows = process.simulate(horizon).to_numpy()
            else:
                first = process.simulate(1, MV=np.asarray(action)[None]).to_numpy()
                tail = process.simulate(horizon - 1).to_numpy()
                rows = np.concatenate((first, tail))
            results.append((rows.tolist(), self.analyser(rows).tolist()))
        return results

    def internal(self, steps):
        rows = self.process.simulate(steps).to_numpy()
        return rows.tolist(), self.analyser(rows).tolist()


class DockerSimulator(simulator.DockerSimulator):
    def __init__(self, image):
        self.worker = subprocess.Popen([
            'docker', 'run', '--rm', '-i', '--platform', 'linux/amd64', '-v', f'{ROOT}:/app:ro',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-w', '/app', image,
            'python', '-u', 'exp_8/simulator.py',
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def nominal(self):
        import numpy as np
        return np.asarray(self.request(op='nominal'), dtype=float)

    def branches(self, actions, horizon):
        return self.request(op='branches', actions=actions, horizon=horizon)

    def internal(self, steps):
        return self.request(op='internal', steps=steps)


def serve():
    import json
    import os
    output = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    backend = NativeSimulator()
    try:
        for line in sys.stdin:
            try:
                message = json.loads(line)
                operation = message.pop('op')
                if operation == 'reset':
                    result = backend.reset(**message).tolist()
                elif operation == 'step':
                    row, value = backend.step(**message)
                    result = [row.tolist(), value]
                elif operation == 'nominal':
                    result = backend.nominal()
                elif operation == 'branches':
                    result = backend.branches(**message)
                elif operation == 'internal':
                    result = backend.internal(**message)
                else:
                    raise ValueError(f'Unknown operation: {operation}')
                response = dict(result=result)
                output.write(json.dumps(response, allow_nan=False) + '\n')
            except Exception as error:
                output.write(json.dumps(dict(error=f'{type(error).__name__}: {error}')) + '\n')
    finally:
        backend.close()


if __name__ == '__main__':
    serve()
