FROM ubuntu:latest

ENV https_proxy http://proxy.ethz.ch:3128/
ENV http_proxy http://proxy.ethz.ch:3128/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y

COPY requirements requirements
RUN pip3 install -r requirements
RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

WORKDIR /EnergyTransitionTasks
