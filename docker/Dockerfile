# Demo docker, not tested for production

# Inspired by
# Detectron2 Dockerfile
# https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile

# TextFuseNet step-by-step installation
# https://github.com/ying09/TextFuseNet/blob/master/step-by-step%20installation.txt 

# Docker conda installation
# https://stackoverflow.com/a/62674910/6760875

FROM nvidia/cuda:10.1-cudnn7-devel

# install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y wget git libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# create environment
RUN conda create --name textfusenet python=3.7.3

# activate environment
SHELL ["conda", "run", "-n", "textfusenet", "/bin/bash", "-c"]

# force enable cuda
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# install packages
RUN conda install pytorch=1.3.1 torchvision cudatoolkit=10.1 -c pytorch
RUN pip install opencv-python tensorboard yacs tqdm termcolor tabulate matplotlib cloudpickle wheel pycocotools

# clone TextFuseNet
RUN git clone https://github.com/ying09/TextFuseNet.git

# set the working directory
WORKDIR TextFuseNet

# install fvcore
RUN pip install fvcore-master.zip

# build TextFuseNet
RUN python setup.py build develop

# activate environment for the user
RUN echo "source activate textfusenet" > ~/.bashrc

# use demo script as an entry point
ENTRYPOINT python demo/${DEMO_FILE} --weights model.pth --output ./output_images/