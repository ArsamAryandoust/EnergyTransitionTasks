FROM ubuntu:latest

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y

RUN pip3 install jupyter
RUN pip3 install -r requirements.txt

WORKDIR /EnergyTransitionTasks

CMD jupyter notebook --ip 0.0.0.0 --port 1111 --no-browser --allow-root
