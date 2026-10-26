# Build from the project root: docker build -f tools/image_to_3d/runpod_service/Dockerfile .
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
ARG CUDA_ARCH=8.9
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} \
    MAX_JOBS=2 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    PYTHONUNBUFFERED=1 \
    PIP_CONSTRAINT=/opt/worker/constraints.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libjpeg-dev libpng-dev libgl1 libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/worker
COPY tools/image_to_3d/runpod_service/constraints.txt /opt/worker/constraints.txt
COPY tools/image_to_3d/runpod_service/source-lock.json /opt/worker/source-lock.json
COPY tools/image_to_3d/runpod_service/build_sources.py /opt/worker/build_sources.py
RUN python build_sources.py
RUN pip install --no-cache-dir ninja cmake packaging setuptools wheel \
    && pip install --no-cache-dir torchvision==0.22.0 \
    && pip install --no-cache-dir -r /opt/Pixal3D/requirements.txt \
    && pip install --no-cache-dir runpod==1.12.0 tensorboard pandas lpips \
    && pip install --no-cache-dir 'natten==0.21.0+torch270cu128' -f https://whl.natten.org \
    && pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1 \
    && pip install --no-cache-dir --no-build-isolation /opt/nvdiffrast /opt/nvdiffrec \
    && pip install --no-cache-dir --no-build-isolation /opt/CuMesh /opt/FlexGEMM \
    && pip install --no-cache-dir --no-build-isolation --no-deps /opt/TRELLIS.2/o-voxel \
    && pip install --no-cache-dir 'https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl'
RUN pip install --no-cache-dir boto3==1.43.118
COPY tools/image_to_3d/runpod_service/storage.py tools/image_to_3d/runpod_service/s3_worker.py tools/image_to_3d/runpod_service/glb.py tools/image_to_3d/runpod_service/protocol.py tools/image_to_3d/runpod_service/model_cache.py tools/image_to_3d/runpod_service/backend.py tools/image_to_3d/runpod_service/handler.py /opt/worker/
ENV PYTHONPATH=/opt/Pixal3D:/opt/worker \
    PIXAL_MODE=multi \
    ATTN_BACKEND=sdpa \
    SPARSE_BACKEND=flex_gemm \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/runpod-volume/pixal3d/cache/huggingface \
    TORCH_HOME=/runpod-volume/pixal3d/cache/torch \
    XDG_CACHE_HOME=/runpod-volume/pixal3d/cache \
    MAX_VIEWS=8 \
    MAX_RESOLUTION=1024
RUN python -c "import runpod, torch, natten, o_voxel, cumesh; import inference_mv"
CMD ["python", "-u", "/opt/worker/handler.py"]
