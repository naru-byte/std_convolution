FROM nvidia/cuda:11.0-devel

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y \
    wget \
    curl \
    build-essential \
    ca-certificates \
    libjpeg-dev \
    ffmpeg \
    cmake \
    libsm6 \
    git \
    less \
    vim

RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build
ENV PATH=$PATH:/opt/conda/bin/

WORKDIR /project

COPY ./conda_environment/environment.yml .
RUN conda env create -f environment.yml && \
    conda clean -afy

COPY . .

# ENTRYPOINT ["/bin/bash", "scripts/docker/run.sh"]

