FROM ubuntu:latest

ENV https_proxy http://proxy.ethz.ch:3128/
ENV http_proxy http://proxy.ethz.ch:3128/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y

RUN pip3 install pandas
RUN pip3 install h5py
RUN pip3 install ase
RUN pip3 install tqdm

ENV PATH="${PATH}:/EnergyTransitionTasks/src"
WORKDIR /EnergyTransitionTasks

