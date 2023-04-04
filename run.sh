#docker build -f Dockerfile -t docker_ett .
docker run -it -v ~/EnergyTransitionTasks:/EnergyTransitionTasks -v ~/../shared_cp:/data -v ~/.aa_secrets:/.aa_secrets docker_ett
