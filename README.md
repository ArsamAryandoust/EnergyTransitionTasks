<img src="https://img.shields.io/badge/datasets-4-blue"/>

# Prediction tasks and datasets for enhancing the global energy transition

Datasets:

* [Building Electricity](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/config_file/BuildingElectricity)
* [Uber Movement](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/config_file/UberMovement)
* [ClimART](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/config_file/ClimART)
* [Open Catalyst](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/config_file/OpenCatalyst)

An effective way to reduce global greenhouse gas emissions and mitigate climate 
change is to electrify most of our energy consumption and supply its electricity 
from renewable sources like wind and solar. In this repository, we provide prediction 
tasks and datasets from different domains such as mobility, material science, 
climate modeling and load forecasting that all contribute to solving the complex 
set of problems involved in enhancing the global energy transition to highly or 
fully renewable systems. 


### Contributing

If you have published or unpublished datasets that fit our purpose, your contribution
is highly appreciated. For changes to code, please download this repository, create 
a new branch, push your changes to the repository, and create a pull request to 
the latest\_release branch.

If you are using the datasets we provide here and have identified any problems 
with these or want to suggest ways to improve these, your feedback is highly 
appreciated. For this, please go to the Discussions tab, create a New discussion,
and give us your detailed feedback.


### Download

Download this repository with all data samples to your home directory:

```
cd 
git clone https://github.com/ArsamAryandoust/EnergyTransitionTasks
cd EnergyTransitionTasks
```


### Docker

Build and run Docker for processing data:
The easiest way to build and run the Docker container is with the `build_and_run.sh` 
script we provide. To do this, execute the following command:

```
./build_and_run.sh
```


### Process data

All raw datasets can be processed using the `main.py` file inside the `src` folder.
All available options can be found by executing the following command inside Docker:
```
python3 src/main.py --help
```

For example, processing the Building Electricity dataset can be done with:
```
python3 src/main.py -building_electricity
```


