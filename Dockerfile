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
        gnutls-bin


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

EXPOSE 7011

RUN python3.6 -m spacy download en 
RUN sh buildproto.sh
CMD ["python3.6" ,"server.py"]
