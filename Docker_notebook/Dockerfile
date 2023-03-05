FROM ubuntu:latest

ENV https_proxy http://proxy.ethz.ch:3128/
ENV http_proxy http://proxy.ethz.ch:3128/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

WORKDIR /EnergyTransitionTasks
CMD jupyter notebook --ip 0.0.0.0 --port 1111 --no-browser --allow-root
