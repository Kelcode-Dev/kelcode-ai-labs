##
## Builder stage (with CUDA headers & build tools)
##
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python, pip, build deps, CUDA headers, Boost, libgomp
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev \
      build-essential \
      libboost-python-dev libboost-thread-dev \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (including pycuda & tensorrt) into /install
WORKDIR /install
COPY requirements.txt .
RUN pip3 install --upgrade pip \
  && pip3 install --no-cache-dir --target=/install -r requirements.txt

# Copy app code into /install
COPY . /install

##
## Final stage (runtime only)
##
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 \
  && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create a non-root user and home
RUN groupadd --gid 1000 appgroup \
 && useradd --uid 1000 --gid appgroup --create-home appuser \
 && mkdir /app \
 && chown appuser:appgroup /app

WORKDIR /app

# Copy in just the installed packages + your code
COPY --chown=appuser:appgroup --from=builder /install /app

USER appuser

EXPOSE 8080

# Launch Uvicorn
ENTRYPOINT ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
