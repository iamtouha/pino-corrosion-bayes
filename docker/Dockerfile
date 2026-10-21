# PINO corrosion surrogate: FEniCSx 0.9 + PyTorch (CUDA) + neuraloperator + Pyro + CalculiX
FROM dolfinx/dolfinx:v0.9.0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        calculix-ccx git less htop && \
    rm -rf /var/lib/apt/lists/*

# Keep dolfinx's numpy; everything else installs against it.
RUN echo "numpy==2.0.2" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt \
        torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -c /tmp/constraints.txt \
        neuraloperator==1.0.2 pyro-ppl==1.9.1 \
        h5py scipy matplotlib pyyaml pandas tqdm pytest requests

ENV PYTHONPATH=/work:/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
WORKDIR /work
CMD ["bash"]
