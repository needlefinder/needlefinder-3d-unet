FROM tensorflow/tensorflow:latest-devel-gpu-py3
MAINTAINER Guillaume Pernelle <gpernelle@gmail.com>

ADD app/requirements.txt /app/
WORKDIR /app

## Clone Needlefinder model
RUN echo '-- Cloning github repository'
RUN pip install -r /app/requirements.txt
RUN ln -s /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
COPY ./app /app
EXPOSE 8888

ENTRYPOINT ["python","/app/fit.py"]
