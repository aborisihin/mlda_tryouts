FROM ubuntu:18.04

ENV LANG=C.UTF-8

# Common packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential && \
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

# Python packages
RUN pip3 install --no-cache-dir --upgrade \
    requests \
    beautifulsoup4 \
    Scrapy \
    pandas \
    scikit-learn \
    hyperopt \
    pymystem3 \
    Flask \
    flask-bootstrap \
    Flask-WTF

# pymystem3 init
RUN python3 -c "import pymystem3; pymystem3.Mystem()"
