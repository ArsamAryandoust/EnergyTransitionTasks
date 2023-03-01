<img src="https://img.shields.io/badge/BuildingElectricity-ready-blue"/> <img src="https://img.shields.io/badge/UberMovement-ready-blue"/> <img src="https://img.shields.io/badge/ClimART-preparing-red"/> <img src="https://img.shields.io/badge/OpenCatalyst-preparing-red"/>

# Prediction tasks and datasets for enhancing the global energy transition

### Datasets

* [Building Electricity](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/BuildingElectricity)
* [Uber Movement](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/UberMovement)
* [ClimART](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/ClimART)
* [Open Catalyst](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/OpenCatalyst)


### Summary
 
Artificial intelligence (AI) and its subfield of machine learning (ML) have a 
significant potential for enhancing technologies, policies and social processes 
that we urgently need for mitigating and adapting to climate change. Improving the 
design of AI and ML models for tackling the energy transition, however, requires 
knowledge about the characteristics of prediction tasks and datasets that are involved. 
The current literature lacks a collection of relevant prediction tasks and their 
datasets, as well as an analysis of their characteristics and relationships to each 
other. In this repository, we review the literature of prediction tasks that are 
relevant for enhancing the global energy transition and process a collection of 
public datasets into a single tabular data format (.csv) for these to simplify the 
design of new AI and ML models for solving them.


### Contributing

* If you have published or unpublished datasets that fit our purpose, your contribution
is highly appreciated. For changes to code, please download this repository, create 
a new branch, push your changes to the repository, and create a pull request to 
the `latest_release` branch.

* If you are using the datasets we provide here and have identified any problems 
with these or want to suggest ways to improve these, your feedback is highly 
appreciated. For this, please go to the `Discussions` tab, create a `New discussion`,
and give us your detailed feedback.


### Download

Download this repository and a sample of each dataset to your home directory:

```
cd 
git clone https://github.com/ArsamAryandoust/EnergyTransitionTasks
cd EnergyTransitionTasks
```


### Docker

The easiest way to build and run a Docker container for processing data is with 
the `build.sh` and `run.sh` scripts we provide. To do this, execute the following 
commands:

```
./build.sh
./run.sh
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


### Tests

The easiest way to build and run Docker containers for integration and unit tests
is through docker compose. To do this, simply execute the following commands:
```
docker-compose build
docker-compose up
```



