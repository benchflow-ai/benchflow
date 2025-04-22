FROM ubuntu:latest

ENV CONDA_ENV_NAME=grader
ENV PYTHON_VERSION=3.12

# Avoid interactive dialog from apt-get and other packages requiring configuration
ENV DEBIAN_FRONTEND=noninteractive

# Install basic packages
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    openssh-server \
    git \
    && rm -rf /var/lib/apt/lists/* # removes cache

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda init

# Create conda environment
RUN /opt/conda/bin/conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
ENV PATH="/opt/conda/bin:${PATH}"

# Install chz from GitHub
RUN /opt/conda/envs/${CONDA_ENV_NAME}/bin/pip install "git+https://github.com/openai/chz.git@97cc0dfb5934a4b99c3a96bdcadcfdbe14812fe8#egg=chz"

# Install paperbench
COPY . /paperbench
RUN /opt/conda/envs/${CONDA_ENV_NAME}/bin/pip install -e /paperbench

WORKDIR /paperbench

RUN mkdir -p /submission /output
