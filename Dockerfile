# Python engine image: train / score / stream / validate / export.
# CPU-only torch keeps the image self-contained and reproducible. This packages
# the tool; how it is invoked (batch CLI now, a streaming runtime later) is a
# separate deployment decision — see reports/PRODUCTION_READINESS.md.
FROM python:3.12-slim-bookworm

# Non-interactive, no bytecode, unbuffered logs (the engine logs JSON to stderr).
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first for layer caching. CPU torch from the dedicated index keeps
# the image lean (no CUDA).
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install numpy scikit-learn onnx onnxruntime

# Project source, then an editable-free install of just this package.
COPY src ./src
COPY benchmarks ./benchmarks
RUN pip install --no-deps .

# Drop root for runtime.
RUN useradd --create-home --uid 10001 idr
USER idr

ENTRYPOINT ["idr-intelligence"]
CMD ["--help"]
