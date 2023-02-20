FROM ubuntu:latest

ENV https_proxy http://proxy.ethz.ch:3128/
ENV http_proxy http://proxy.ethz.ch:3128/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV PATH="${PATH}:/EnergyTransitionTasks/src"
WORKDIR /EnergyTransitionTasks
