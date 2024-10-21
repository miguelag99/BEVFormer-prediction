# FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Obtain the UID and GID of the current user to create a user with the same ID, this is to avoid permission issues when mounting local volumes.
ARG USER
ARG USER_ID
ARG USER_GID

ENV FORCE_CUDA="1" 

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add user.
RUN groupadd -g $USER_GID $USER \
    && useradd --uid $USER_ID --gid $USER_GID -m $USER \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

RUN apt install sudo libgl1-mesa-glx mesa-utils libglapi-mesa libqt5gui5 -y
RUN apt-get install -y build-essential cmake git curl ca-certificates \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    wget \
    jupyter \
    tmux

RUN python -m pip install --upgrade pip
ENV PATH="${PATH}:/home/$USER/.local/bin"

RUN mkdir /.cache
COPY requirements.txt /.cache

USER $USER

# Install requirements
RUN pip install -r /.cache/requirements.txt

# Install mmcv from source
WORKDIR /home/$USER
RUN git clone https://github.com/open-mmlab/mmcv.git
WORKDIR /home/$USER/mmcv
RUN pip install -e . -v

# Disable jupyter authentication
WORKDIR /home/$USER
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py


WORKDIR /home/$USER/Datasets
WORKDIR /home/$USER/workspace
ENTRYPOINT [ "/bin/bash", "-l", "-c"  ]