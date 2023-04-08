# Pull latest code from EnergyTransitionTasks
git stash push .
git pull origin latest_release

# Clone code from selberai branch latest_release
sudo rm -r src/selberai
git clone -b latest_release https://github.com/Selber-AI/selberai
mv selberai/selberai src/
sudo rm -r selberai

# Start docker container
#docker build -f Dockerfile -t docker_ett .
docker run -it -v ~/EnergyTransitionTasks:/EnergyTransitionTasks -v ~/../shared_cp:/data -v ~/.aa_secrets:/.aa_secrets docker_ett
