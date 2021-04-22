FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3.6 \
        python3.6-dev \
        python3-pip \
        python-setuptools \
        cmake \
        wget \
        curl \
        libsm6 \
        libxext6 \ 
        libxrender-dev \
        vim \
        lsof

RUN SNETD_GIT_VERSION=`curl -s https://api.github.com/repos/singnet/snet-daemon/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")' || echo "v3.1.6"` && \
    SNETD_VERSION=${snetd_version:-${SNETD_GIT_VERSION}} && \
    cd /tmp && \
    wget https://github.com/singnet/snet-daemon/releases/download/${SNETD_VERSION}/snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    tar -xvf snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    mv snet-daemon-${SNETD_VERSION}-linux-amd64/snetd /usr/bin/snetd && \
    rm -rf snet-daemon-*


RUN python3.6 -m pip install -U pip
RUN python3.6 -m pip install --upgrade setuptools

COPY requirement.txt /tmp

WORKDIR /tmp

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


RUN python3.6 -m pip install -r requirement.txt

COPY . /fake-news-classification 

WORKDIR /fake-news-classification/service

#RUN git clone https://gitlab.com/nunet/fake-news-detection/data-storage

#RUN cp data-storage/fake-news-detector.pth  .

# VOLUME /image-retrieval-in-pytorch/data/classed_data

EXPOSE 7010
EXPOSE 7011

RUN python3.6 -m spacy download en 
RUN sh buildproto.sh

WORKDIR /fake-news-classification/
CMD ["python3.6" ,"service/server.py"]
