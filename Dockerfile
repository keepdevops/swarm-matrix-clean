# Containerized matrix-safe for the PlantUML editor's `--profile llm`.
#
# Uses the in-process `llama_cpp_python` backend (CPU) so no llama-server binary
# is baked in. Model weights are MOUNTED read-only at /models, never in the image
# (MATRIX_SAFE_MODELS_DIR=/models; agent configs reference paths relative to it).
#
#   docker build -t matrix-safe:local .
FROM python:3.12-slim

# Build tools so llama-cpp-python can compile from source if no prebuilt wheel
# matches the target platform (e.g. linux/arm64).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "uvicorn[standard]>=0.29" \
 && pip install --no-cache-dir "llama-cpp-python>=0.2.90" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Only what the FastAPI server needs (keeps the image small; skips frontend/tests).
COPY server/ server/
COPY backends/ backends/
COPY orchestration/ orchestration/
COPY config/ config/

ENV MATRIX_SAFE_MODELS_DIR=/models
EXPOSE 8765

# Bind 0.0.0.0 (server/__main__ binds 127.0.0.1, unreachable across containers).
CMD ["uvicorn", "server.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765"]
