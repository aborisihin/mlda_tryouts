FROM ubuntu:18.04

ENV LANG=C.UTF-8

# Common packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        swig \
        git \
        zip && \
    rm -rf /var/lib/apt/lists/*

# Python 3.6
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3.6 \
        python3.6-dev \
        python3.6-venv && \
    pip3 install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# auto-sklearn initialization
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt \
  | xargs -n 1 -L 1 pip3 installw

# Python packages
RUN pip3 install --no-cache-dir --upgrade \
    pandas \
    jupyter \
    auto-sklearn
