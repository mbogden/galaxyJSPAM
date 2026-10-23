# E6a: GPU image for train-fitness-score.py
# Same TensorFlow as 2023 (2.9.1, CUDA 11.2 / cuDNN 8.1) plus the packages the script needs,
# pinned to the versions in the legacy ogdenm12/beta-2 image.
# Build:  docker build -t e6a-tf291-gpu docker/
FROM tensorflow/tensorflow:2.9.1-gpu
# CUDA 11.2's default allocator fails on this laptop (WSL2, driver 621 / CUDA 13.5):
# every cudaMalloc reports OUT_OF_MEMORY and cuBLAS won't start. The async allocator works.
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
RUN pip install --no-cache-dir \
        opencv-python-headless==4.6.0.66 \
        matplotlib==3.6.2 \
        keras-tuner==1.3.0 \
        scipy==1.10.1
