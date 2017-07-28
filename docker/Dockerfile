FROM ubuntu:16.04

MAINTAINER Guillaume Pernelle <gpernelle@gmail.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYENV_DIR /root/.pyenv
ENV PATH $PYENV_DIR/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libhdf5-dev g++ graphviz libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion curl libcurl3 libcurl3-dev

RUN curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
    pyenv init - && \
    pyenv virtualenv-init - && \
    pyenv update && \
    pyenv install anaconda3-4.4.0 && \
    pyenv global anaconda3-4.4.0

RUN useradd -m deepinfer && echo "deepinfer:deepinfer" | chpasswd && adduser deepinfer sudo

RUN mkdir -p $PYENV_DIR && \
    chown deepinfer $PYENV_DIR -R

USER deepinfer

RUN mkdir -p /home/deepinfer/github && \
    mkdir /home/deepinfer/data && \
    cd /home/deepinfer/github/ && \
    git clone https://github.com/needlefinder/needlefinder-3d-unet && \
    cd needlefinder-3d-unet

# Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install ipdb pytest pytest-cov python-coveralls coverage==3.7.1 pytest-xdist pep8 pytest-pep8 pydot_ng && \
    conda clean -yt

EXPOSE 8888

WORKDIR "/home/deepinfer"
# CMD ["/bin/bash"]
ENTRYPOINT ["python","/home/deepinfer/github/needlefinder-3d-unet/fit.py"]
