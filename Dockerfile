FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 AS base-image

# Install Python and its tools
RUN apt update && apt install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade
# Install all basic packages
RUN apt-get install -y apt-utils
RUN apt-get install -y curl unzip vim wget
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN pip3 install \
    # Jupyter itself
    jupyter \
    # Numpy and Pandas are required a-priori
    numpy pandas \
    # PyTorch with CUDA 10.1 support and Torchvision
    torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html \
    # Upgraded version of Tensorboard with more features
    tensorboard \
    # matplotlib for making plots
    matplotlib \
    # scikit-learn to access some  metrics
    scikit-learn \
    # OpenCV standard computer vision library
    opencv-python


RUN pip3 install --upgrade pip setuptools

FROM base-image
# Install additional packages
RUN pip3 install \
    # comet logger
    comet_ml \
    # Progress bar to track experiments
    tqdm \
    # Pytorch lightning
    pytorch-lightning==1.1.6 \
    # tensorboard logger
    tensorboard_logger
