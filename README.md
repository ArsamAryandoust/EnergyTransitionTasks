# A collection of datasets and prediction tasks for enhancing the global energy transition

An effective way to reduce global greenhouse gas emissions and mitigate climate 
change is to electricity most of our energy consumption and supply its electricity 
from renewable energy sources like wind and solar. In this repository, we provide 
a number of prediction tasks and datasets from different domains such as mobility, 
climate modeling and material science that all contribute to solving the complex 
set of problems involved in enhancing the global energy transition towards highly 
renewable electricity systems.




### UberMovement

The Uber Movement project provides travel time data between different city zones
in the form of origin-destination matrices for about 50 cities around the world 
under a CC BY-NC 3.0 US license (https://creativecommons.org/licenses/by-nc/3.0/us/). 
Travel time data can be used, among many other things in planning urban mobility, 
to infer where cars are parked at what times throughout a day, week, season and 
year in a city [1]. This information is important for planning and operating power 
systems in places where we are transitioning to using electric cars. For example, 
it will allow us to know how much power must flow to which parts of a city when 
electricity generation from solar power is to be used to charge electrc vehicles 
during mid-day.


### ClimART

The ClimART project provides data on the inputs and outputs of the Canadian Earth
System Model for emulating atmospheric radiative transfer between the years 1850 
and 2100 under a CC BY 4.0 license (https://creativecommons.org/licenses/by/4.0/).
The computation of atmospheric radiative transfer is the computational bottleneck 
for current weather and climate models due to its large complexity. Training a
machine learning model that is able to predict such model outputs at faster 
computational speed can therefore fill the information gaps for times and places
where traditional model results are too computationaly expensive to perform.


### References

[1] Aryandoust A., Van Vliet O. & Patt A. City-scale car traffic and parking density 
maps from Uber Movement travel time data, Scientific Data 6, 158 (2019). 
https://doi.org/10.1038/s41597-019-0159-6

[2] Cachay, S. R., Ramesh, V., Cole, J. N. S., Barker, H. & Rolnick D. ClimART:
A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and 
Climate Models.


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
