FROM nvidia/cuda:11.0.3-base-ubuntu20.04 
# Source: https://github.com/uoe-agents/epymarl/blob/main/docker/Dockerfile
# Note: I needed to replace the image as it was removed

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
# RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
#     rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Python packages we use (or used at one point...)
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger
RUN pip3 install pygame

RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch
RUN pip3 install torchvision snakeviz pytest probscale

## -- SMAC
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /pymarl/3rdparty/StarCraftII


#### -------------------------------------------------------------------
#### new for the simple-es extension
#### -------------------------------------------------------------------

# mpe
WORKDIR /project
RUN apt update && apt install -y git
RUN git clone https://github.com/semitable/multiagent-particle-envs.git; 
RUN cd multiagent-particle-envs && pip install -e .

# my project
COPY . /project/es
WORKDIR /project/es/
RUN pip install swig 
RUN pip install -r requirements.txt
RUN ln -s /usr/bin/python3 /usr/bin/python

# additional dependencies
RUN apt install -y zip

