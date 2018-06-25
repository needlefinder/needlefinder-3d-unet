FROM tensorflow/tensorflow:latest-devel-py3
MAINTAINER Guillaume Pernelle <gpernelle@gmail.com>

ADD app/requirements.txt /app/
WORKDIR /app

## Clone Needlefinder model
RUN echo '-- Cloning github repository'
RUN pip install -r /app/requirements.txt
COPY ./app /app
EXPOSE 8888

ENTRYPOINT ["python","/app/fit.py"]
