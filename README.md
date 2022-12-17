# A collection of datasets and prediction tasks for enhancing the global energy transition

An effective way to reduce global greenhouse gas emissions and mitigate climate 
change is to electricity most of our energy consumption and supply its electricity 
from renewable energy sources like wind and solar. In this repository, we provide 
a number of prediction tasks and datasets from different domains such as mobility, 
climate modeling and material science that all contribute to solving a complex 
set of problems involved in enhancing the global energy transition towards highly 
renewable electricity systems.




### UberMovement

The Uber Movement project provides travel time data between different city zones
for more than 50 cities around the globe under a CC BY-NC 3.0 US license 
(https://creativecommons.org/licenses/by-nc/3.0/us/). Travel time data is important
for many applications of planning urban mobility. In the light of the energy 
transition, a recent study shows how Uber Movement travel time data can be used 
to estimate car parking density maps. These maps allow local distribution operators
for electricity and local city governance to better understand where cars are 
parked at what times. This then shows what it requires in terms of power flows and
the distribution of electric charging stations for electrifying urban mobility. 



## Download
Download this repository to your home directory:

```
cd 
git clone https://github.com/ArsamAryandoust/TasksEnergyTransition
cd TasksEnergyTransition
```


## Jupyter notebooks inside Docker containers

Build Jupyter notebook container:

```
docker build -t main_notebook DockerNotebook
```


Start container with CPU only:

```
docker run -it -v ~/TasksEnergyTransition:/TasksEnergyTransition -p 3333:1111 main_notebook
```

Open the link that shows in your terminal with a browser. Then, replace the port 
1111 with 3333 in your browser link to see notebooks inside the Docker container.
