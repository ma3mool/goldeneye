FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
RUN apt update && apt-get install -y \
    ninja-build \
    python3.6 \
    python3-pip \
    git
RUN pip3 install tqdm pytest torch qtorch numpy
RUN git clone https://github.com/ma3mool/goldeneye.git
RUN mkdir datasets
ENV ML_DATASETS=/datasets/imagenet/

