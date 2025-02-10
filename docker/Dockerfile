FROM debian:11

RUN apt-get update && \
    apt-get install -y \
    curl \
    git \
    gnupg2 \
    vim \
    wget

RUN apt-get update && \
    apt-get install -y \
    python3 python3-dev python3-venv python3-distutils python3-pip

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update && \
    apt-get install -y \
    gasket-dkms libedgetpu1-std

RUN apt-get install -y python3-tflite-runtime
RUN apt-get install -y python3-pycoral


RUN apt-get install -y pciutils

