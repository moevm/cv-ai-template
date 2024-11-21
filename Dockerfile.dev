FROM nvcr.io/nvidia/pytorch:23.12-py3
ENV DEBIAN_FRONTEND=noninteractive

# Setup non-root user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install ROS2 humble
RUN apt update || true && apt install -q -y --no-install-recommends \
    bash-completion \
    gnupg2 \
    cmake \
    curl

RUN pip install -U pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN echo "sudo ldconfig" >> /home/${USERNAME}/.bashrc

USER $USERNAME
CMD ["bash"]

