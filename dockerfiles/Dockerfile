FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /workspace/

COPY requirements.txt requirements.txt

ENV DEBIAN_FRONTEND nointeractive
ENV LC_ALL C.UTF-8
ENV TZ Asia/Tokyo
ENV PYTHONPATH $PYTHONPATH:/workspace/

RUN apt update \
    && apt upgrade -y \
    && apt install -y --no-install-recommends python3-pip python-is-python3 git \
    && pip install --upgrade pip

RUN pip install -r requirements.txt