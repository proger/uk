FROM kaldiasr/kaldi:latest

RUN apt-get update && apt-get -y install python3-pip cmake
RUN pip3 install --upgrade pip
RUN ln -sf /opt/kaldi /kaldi
WORKDIR /uk
COPY README.md README.md
COPY setup.py setup.py
RUN pip3 install --no-cache-dir .
COPY . .
