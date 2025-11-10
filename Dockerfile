# Use Ubuntu 22.04 as base instead of 20.04
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    echo "UTC-08" > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    gcc \
    g++ \
    make \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    r-base \
    vcftools \
    plink \
    graphviz \
    default-jre-headless \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O /tmp/anaconda.sh \
    && bash /tmp/anaconda.sh -b -p /opt/anaconda3 \
    && rm /tmp/anaconda.sh

# Add Anaconda to PATH
ENV PATH="/opt/anaconda3/bin:$PATH"

# Add conda channels
RUN conda config --system --add channels anaconda && \
    conda config --system --add channels conda-forge && \
    conda config --system --add channels bioconda && \
    conda config --system --add channels defaults && \
    conda config --system --add channels nvidia && \
    conda config --system --add channels pytorch && \
    conda config --system --set channel_priority flexible

# Create conda environment
RUN conda create -y --name maize python=3.10
RUN conda init bash

# Install Python packages in conda environment
RUN conda run -n maize conda install -y \
    asttokens backcall brotli-bin bzip2 ca-certificates certifi charset-normalizer \
    comm cycler decorator entrypoints executing icu idna ipython jedi jinja2 \
    jupyter_client jupyter_core kiwisolver libblas libbrotlicommon libbrotlidec \
    libbrotlienc libcblas libffi liblapack libopenblas libpng libsodium libsqlite \
    libzlib matplotlib matplotlib-base matplotlib-inline nest-asyncio parso patsy \
    pexpect pickleshare pip platformdirs pooch prompt-toolkit psutil ptyprocess \
    pure_eval pygments pyparsing pysocks python python-dateutil python-tzdata python_abi \
    pyzmq requests seaborn seaborn-base setuptools six stack_data statsmodels tabulate \
    tk tornado traitlets typing_extensions urllib3 wcwidth wheel xz zeromq \
    && conda clean -a

# Pip installs in conda environment
RUN conda run -n maize pip install \
    alembic cmaes colorlog greenlet importlib-metadata importlib-resources \
    joblib lightgbm mako markupsafe numpy optuna packaging pandas pyarrow pytz \
    pyyaml scikit-learn sqlalchemy threadpoolctl tqdm typing-extensions zipp \
    && conda clean -a

# Install R packages
RUN R -e "install.packages(c('arrow', 'data.table', 'AGHmatrix', 'devtools', 'asreml'))"
RUN R -e "install.packages(c('vcfR', 'plink'))"

# Add Pegasus APT repo and HTCondor repo
RUN curl https://download.pegasus.isi.edu/pegasus/gpg.txt | apt-key add - \
&& echo 'deb https://download.pegasus.isi.edu/pegasus/ubuntu noble main' >/etc/apt/sources.list.d/pegasus.list \
&& apt-get update \
&& apt-get install pegasus

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
