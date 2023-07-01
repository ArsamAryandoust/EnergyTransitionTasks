# BE
./run_main.sh -baseline BuildingElectricity buildings_92 1.0
./run_main.sh -baseline BuildingElectricity buildings_451 0.1

# Polianna
./run_main.sh -baseline Polianna article_level 1.0

# WF
./run_main.sh -baseline WindFarm days_177 0.1
./run_main.sh -baseline WindFarm days_245 0.1

# UM
./run_main.sh -baseline UberMovement cities_10 1.0
./run_main.sh -baseline UberMovement cities_20 0.005
./run_main.sh -baseline UberMovement cities_43 0.001

# ClimART
./run_main.sh -baseline ClimART pristine 0.1
./run_main.sh -baseline ClimART clear_sky 0.05

# OpenCatalyst
./run_main.sh -baseline OpenCatalyst oc20_s2ef 1.0
./run_main.sh -baseline OpenCatalyst oc20_is2re 1.0
./run_main.sh -baseline OpenCatalyst oc22_s2ef 1.0
./run_main.sh -baseline OpenCatalyst oc22_is2re 1.0


