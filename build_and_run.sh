

#docker build DockerNotebook -t main_notebook
#docker run -it -v ~/EnergyTransitionTasks:/EnergyTransitionTasks -p 3333:1111 main_notebook


docker build Docker -t main
docker run -it -v ~/EnergyTransitionTasks:/EnergyTransitionTasks main
