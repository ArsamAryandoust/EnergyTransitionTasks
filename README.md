<img src="https://img.shields.io/badge/datasets-4-blue"/>

# Prediction tasks and datasets for enhancing the global energy transition

### Datasets

* [Building Electricity](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/BuildingElectricity)
* [Uber Movement](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/UberMovement)
* [ClimART](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/ClimART)
* [Open Catalyst](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/OpenCatalyst)

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

If you have published or unpublished datasets that fit our purpose, your contribution
is highly appreciated. For changes to code, please download this repository, create 
a new branch, push your changes to the repository, and create a pull request to 
the `latest_release` branch.

If you are using the datasets we provide here and have identified any problems 
with these or want to suggest ways to improve these, your feedback is highly 
appreciated. For this, please go to the `Discussions` tab, create a `New discussion`,
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


