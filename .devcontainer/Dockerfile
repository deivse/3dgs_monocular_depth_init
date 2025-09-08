FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

ARG CUDA_VERSION=12.1

RUN sudo apt-get update && sudo apt-get -y install ninja-build

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
RUN sudo dpkg -i cuda-keyring_1.0-1_all.deb
RUN sudo apt-get update
RUN sudo apt-get -y install cuda-$CUDA_VERSION
RUN rm cuda-keyring_*.deb

RUN ls /usr/local/cuda-$CUDA_VERSION/
ENV CUDA_HOME=/usr/local/cuda-$CUDA_VERSION

SHELL ["/bin/bash", "--login", "-c"]

RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b -f
RUN rm Miniforge3-$(uname)-$(uname -m).sh

ENV CONDA_HOME="/root/miniforge3"
RUN "$CONDA_HOME/condabin/conda" init bash zsh

# Install dependencies
COPY ./environment.yml /tmp/environment.yml
RUN conda update -n base -c conda-forge conda
RUN conda env create --file /tmp/environment.yml

# Install direnv
RUN sudo apt-get -y install direnv
RUN echo 'eval "$(direnv hook bash)"' >> ~/.bashrc

# Allow ssh
RUN apt-get update && apt-get install -y --no-install-recommends net-tools iputils-ping openssh-client openssh-server 


RUN mkdir /var/run/sshd
RUN echo 'root:test' | chpasswd
RUN sed -i 's/.*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
