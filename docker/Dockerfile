FROM ubuntu:16.04

MAINTAINER Guillaume Pernelle <gpernelle@gmail.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libhdf5-dev g++ graphviz libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion curl libcurl3 libcurl3-dev

RUN useradd -m deepinfer

WORKDIR /home/deepinfer
USER deepinfer

ENV HOME  /home/deepinfer
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
    pyenv init - && \
    pyenv virtualenv-init - && \
    pyenv update && \
    pyenv install anaconda3-4.4.0
    
RUN pyenv global anaconda3-4.4.0
RUN pyenv rehash

# Clone Needlefinder model
RUN mkdir -p /home/deepinfer/github && \
    mkdir /home/deepinfer/data && \
    cd /home/deepinfer/github/ && \
    git clone https://github.com/needlefinder/needlefinder-3d-unet && \
    cd needlefinder-3d-unet

# Python
RUN pyenv global anaconda3-4.4.0 && \
    pip install --upgrade pip && \
    pip install -r /home/deepinfer/github/needlefinder-3d-unet/requirements.txt


EXPOSE 8888

ENTRYPOINT ["python","/home/deepinfer/github/needlefinder-3d-unet/fit.py"]
