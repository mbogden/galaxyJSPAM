# E1 base-image experiment: CUDA 12.6 runtime + Python/uv + OpenMPI + PyTorch + Keras 3 + gfortran.
# Build (from the repo root):  docker build -t e1-cuda:dev docker/e1
# The final project Dockerfile will install from uv.lock instead of the pins below.

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=torch \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# Ubuntu 24.04 packages: Python 3.12, compilers (mpi4py and f2py builds), OpenMPI 4.1.6 (matches Babbage).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        build-essential gfortran pkg-config \
        openmpi-bin libopenmpi-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

RUN uv venv /opt/venv --python /usr/bin/python3 \
    && uv pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1 \
    && uv pip install numpy==2.5.4 keras==3.15.1 mpi4py==4.1.2 meson ninja

COPY env_check.py mpi_check.py fortran_check.sh run_all.sh /opt/e1/
