# The service on an NVIDIA GPU: CUDA 12.4, PyTorch 2.4.1, and gsplat 1.5.3 from its
# prebuilt wheels, so nothing is compiled here or at start. gsplat publishes those
# wheels for PyTorch 2.4 on Python 3.10, which is Ubuntu 22.04's.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# The slow layers first, so a code change rebuilds in seconds.
RUN pip3 install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install "gsplat==1.5.3+pt24cu124" --extra-index-url https://docs.gsplat.studio/whl/pt24cu124

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip3 install .

ENV DATA_DIR=/data
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["python3", "-m", "vehicle_splat", "serve", "--host", "0.0.0.0", "--port", "8000"]
