#FROM kaixhin/cuda-theano:latest
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04
MAINTAINER Paolo Zaffino <p.zaffino@unicz.it>

# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
RUN apt-get update
RUN apt-get install -y \
  gfortran \
  git \
  wget \
  liblapack-dev \
  libopenblas-dev \
  python-dev \
  python-pip \
  python-nose 

# Set CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin

# Install CMake 3
RUN cd /root && wget http://www.cmake.org/files/v3.8/cmake-3.8.1.tar.gz && \
  tar -xf cmake-3.8.1.tar.gz && cd cmake-3.8.1 && \
  ./configure && \
  make -j "$(nproc)" && \
  make install

# Install Cython
RUN pip install Cython numpy scipy

# Clone libgpuarray repo and move into it
RUN cd /root && git clone https://github.com/Theano/libgpuarray.git && cd libgpuarray && \
# Make and move into build directory
  mkdir Build && cd Build && \
# CMake
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr && \
# Make
  make -j"$(nproc)" && \
  make install
# Install pygpu
RUN cd /root/libgpuarray && \
  python setup.py build_ext -L /usr/lib -I /usr/include && \
  python setup.py install

RUN pip install --upgrade six
# Install bleeding-edge Theano
#RUN pip install --upgrade pip
RUN easy_install -U pip
#RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
#RUN pip install --upgrade --no-deps https://github.com/Theano/Theano.git
RUN pip install --upgrade --no-deps theano
# Set up .theanorc for CUDA
RUN echo "[global]\ndevice=cuda\nfloatX=float32\noptimizer_including=cudnn\n[lib]\ncnmem=0.1\n[nvcc]\nfastmath=True" > /root/.theanorc

RUN pip install matplotlib scikit_learn SimpleITK lasagne

ADD app/requirements.txt /app/
WORKDIR /app

## Clone Needlefinder model
#RUN echo '-- Cloning github repository'
#RUN pip install -r /app/requirements.txt
#RUN ln -s /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
COPY ./app /app
EXPOSE 8888

ENTRYPOINT ["python","/app/fit.py"]
