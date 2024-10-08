from mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

RUN sudo apt update
RUN pipx install poetry

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
RUN sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
RUN sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
RUN sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN sudo apt-get update
RUN sudo apt-get -y install cuda-toolkit-12-6

RUN echo "export VIRTUALENV_SYSTEM_SITE_PACKAGES=true" >> ~/.bashrc
RUN echo "export VIRTUALENV_SYSTEM_SITE_PACKAGES=true" >> ~/.zshrc

RUN sudo apt-get install -y python3-pip
RUN /usr/bin/python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
