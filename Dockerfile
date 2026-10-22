# The simulator binary is an x86-64 Linux ELF shared library.
# On Apple Silicon, build/run this image with: --platform linux/amd64
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY temain.py libtemain.so generate.py experiment_paths.py ./

RUN pip install --no-cache-dir numpy pandas

# A small smoke test; override this command for experiments.
CMD ["python", "-c", "from temain import TemainProcess; print(TemainProcess(freq='60s').simulate(1))"]
