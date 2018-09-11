FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
MAINTAINER Paolo Zaffino <p.zaffino@unicz.it>

# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
ENV TZ=Europe/Rome
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata 

RUN apt-get install -y \
  gfortran \
  git \
  wget \
  cmake \
  liblapack-dev \
  libopenblas-dev \
  python-dev \
  python-pip \
  python-nose \
  python-numpy \ 
  python-scipy \ 
  python-matplotlib

# Set CUDA_ROOT and cudnn stuff
ENV CUDA_ROOT /usr/local/cuda/bin

RUN cd /usr/local/cuda/include
RUN cp /usr/include/cudnn.h .
RUN cd /usr/local/cuda/lib64
RUN cp /usr/lib/x86_64-linux-gnu/libcudnn* .

# Install Cython
RUN pip install Cython
RUN pip install scikit_learn SimpleITK

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

# Install theano and lasagne
RUN pip install --upgrade six
RUN pip install --upgrade pip
RUN pip install --upgrade https://github.com/Theano/Theano/archive/master.zip 
# Set up .theanorc for CUDA
RUN echo "[global]\ndevice=cuda\nfloatX=float32\n[nvcc]\nfastmath=True\n[cuda]\nroot=/usr/local/cuda/bin\n[dnn]\ninclude_path=/usr/local/cuda/include\nlibrary_path=/usr/local/cuda/lib64\n" > /root/.theanorc
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

# Add needlefinder files
ADD app/requirements.txt /app/
WORKDIR /app
COPY ./app /app
RUN echo "[global]\ndevice=cpu\nfloatX=float32\n[nvcc]\nfastmath=True\n[cuda]\nroot=/usr/local/cuda/bin\n[dnn]\ninclude_path=/usr/local/cuda/include\nlibrary_path=/usr/local/cuda/lib64\n" > /root/.theanorc
RUN THEANO_FLAGS="device=cpu, dnn.base_path=/usr/local/cuda" python /app/compile_and_save_function.py
RUN echo "[global]\ndevice=cuda\nfloatX=float32\n[nvcc]\nfastmath=True\n[cuda]\nroot=/usr/local/cuda/bin\n[dnn]\ninclude_path=/usr/local/cuda/include\nlibrary_path=/usr/local/cuda/lib64\n" > /root/.theanorc
EXPOSE 8888

ENTRYPOINT ["python","/app/fit.py"]

