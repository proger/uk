FROM kaldiasr/kaldi:latest

RUN apt-get update && apt-get -y install python3-pip
RUN ln -sf /opt/kaldi /kaldi
WORKDIR /uk
COPY setup.py setup.py
RUN pip3 install .
COPY . .
