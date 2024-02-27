while getopts ubn flag
do
  case "${flag}" in
    u) update=1;;
    b) build=1;;
    n) notebook=1;;
  esac
done


if [ $update -eq 1 ]; then
  # Pull latest code from EnergyTransitionTasks
  git stash push .
  git pull origin lead_dev

  # Clone code from selberai branch lead_dev
  sudo rm -r src/selberai
  git clone -b lead_dev https://github.com/Selber-AI/selberai
  mv selberai/selberai src/
  sudo rm -r selberai

  # Clonde code from open catalyst project
  sudo rm -r src/ocpmodels
  git clone https://github.com/Open-Catalyst-Project/ocp
  mv ocp/ocpmodels src/
  sudo rm -r ocp
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





