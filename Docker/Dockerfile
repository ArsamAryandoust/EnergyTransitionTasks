FROM ubuntu:latest

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y

RUN pip3 install jupyter
RUN pip3 install pandas
RUN pip3 install h5py
RUN pip3 install ase
RUN pip3 install tqdm

WORKDIR /EnergyTransitionTasks/src

CMD python3 main.py
