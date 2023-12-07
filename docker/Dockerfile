ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV TZ=Asia/Kolkata DEBIAN_FRONTEND=noninteractive
# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools
# Install MMEngine and MMCV
RUN pip install openmim
RUN mim install mmengine
RUN mim install "mmpose==0.28.1"
RUN mim install "mmcv-full==1.5.3"
RUN pip install -U torchmetrics timm
RUN pip install numpy scipy --upgrade
RUN pip install future tensorboard

WORKDIR PoseAnything

COPY models PoseAnything/models
COPY configs PoseAnything/configs
COPY pretrained PoseAnything/pretrained
COPY requirements.txt PoseAnything/
COPY tools PoseAnything/tools
COPY setup.cfg PoseAnything/
COPY setup.py PoseAnything/
COPY test.py PoseAnything/
COPY train.py PoseAnything/
COPY README.md PoseAnything/

RUN mkdir -p PoseAnything/data/mp100
WORKDIR PoseAnything

# Install MMPose
RUN conda clean --all
ENV FORCE_CUDA="1"
RUN python setup.py develop