<img src="https://img.shields.io/badge/Building_Electricity-preparing-red"/> <img src="https://img.shields.io/badge/Wind_Farm-preparing-red"/> <img src="https://img.shields.io/badge/Generation_Scheduling-preparing-red"/> <img src="https://img.shields.io/badge/Uber_Movement-preparing-red"/> <img src="https://img.shields.io/badge/ClimART-preparing-red"/> <img src="https://img.shields.io/badge/Climate_Downscaling-preparing-red"/> <img src="https://img.shields.io/badge/Open_Catalyst-preparing-red"/> <img src="https://img.shields.io/badge/Polianna-preparing-red"/>

# 1X machine learning tasks and datasets for enhancing the global energy transition

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


## Datasets

* [Building Electricity](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/BuildingElectricity)
* [Wind Farm](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/WindFarm)
* [Generation Scheduling](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/GenerationScheduling)
* [Uber Movement](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/UberMovement)
* [ClimART](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/ClimART)
* [Climate Downscaling](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/ClimateDownscaling)
* [Open Catalyst](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/OpenCatalyst)
* [Polianna](https://github.com/ArsamAryandoust/EnergyTransitionTasks/tree/master/datasets/Polianna)


## Download Code
Download this repository to your home directory:

```
cd 
git clone https://github.com/ArsamAryandoust/EnergyTransitionTasks
cd EnergyTransitionTasks
```

## Run Docker
The easiest way to build and run a Docker container for downloading, processing
and uploading datasets is with the `run.sh` script we provide. To do this, 
execute the following command to build and run a Docker container:

```
./run.sh
```

All commands are implemented through the `main.py` file inside the `src` folder.
Make sure to comment out the building commands inside `run.sh` after the first
execution, so you don't rebuild redundant containers.


## Download datasets
Any dataset can be downloaded using the `-download` flag followed by the 
<dataset name>:

```
python3 src/main.py -download <dataset name>
```

Any dataset can be downloaded using the `-download` flag followed by the 
<dataset name>:


## Examples


## Contributing

* If you have published or unpublished datasets that fit our purpose, your 
contribution is highly appreciated. For changes to code, please download this 
repository, create a new branch, push your changes to the repository, and 
create a pull request to the `latest_release` branch.

* If you are using the datasets we provide here and have identified any 
problems with these or want to suggest ways to improve these, your feedback is 
highly appreciated. For this, please go to the `Discussions` tab, create a 
`New discussion`, and give us your detailed feedback.



