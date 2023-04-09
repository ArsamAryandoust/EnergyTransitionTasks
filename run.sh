while getopts ubn flag
do
  case "${flag}" in
    u) update=1;;
    b) build=1;;
    n) notebook=1;;
  esac
done


if [ $update -eq 1 ]; then
  # save copy notebooks
  sudo cp -r notebooks .notebooks
  
  # Pull latest code from EnergyTransitionTasks
  git stash push .
  git pull origin latest_release
  
  # reload notebooks
  sudo cp -r .notebooks notebooks
 
  # Clone code from selberai branch latest_release
  sudo rm -r src/selberai
  git clone -b latest_release https://github.com/Selber-AI/selberai
  mv selberai/selberai src/
  sudo rm -r selberai
fi


if [ $build -eq 1 ]; then
  # build docker container
  docker build -f Dockerfile -t docker_ett .
fi


if [ $notebook -eq 1 ]; then
  # start notebook container
  docker run -it -p 1111:1111 -v ~/EnergyTransitionTasks:/EnergyTransitionTasks -v ~/../shared_cp:/data -v ~/.aa_secrets:/.aa_secrets docker_ett
else
  # start normal container
  docker run -it -v ~/EnergyTransitionTasks:/EnergyTransitionTasks -v ~/../shared_cp:/data -v ~/.aa_secrets:/.aa_secrets docker_ett
fi





