FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        swig \
        cmake \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        git \
        curl \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --upgrade pip

RUN pip install "gymnasium[box2d]"
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
    pygame \
    numpy \
    pandas \
    matplotlib \
    opencv-python \
    pyyaml

WORKDIR /workspace
COPY . /workspace

CMD ["bash"]
