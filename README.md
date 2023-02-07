<img src="https://img.shields.io/badge/datasets-4-blue"/>

# Prediction tasks and datasets for enhancing the global energy transition


An effective way to reduce global greenhouse gas emissions and mitigate climate 
change is to electrify most of our energy consumption and supply its electricity 
from renewable sources like wind and solar. In this repository, we provide prediction 
tasks and datasets from different domains such as mobility, material science, 
climate modeling and load forecasting that all contribute to solving the complex 
set of problems involved in enhancing the global energy transition to highly or 
fully renewable systems. 


<img src="/figures/data_format.png" />


All data is prepared in the above tabular data format as a .csv file, where each 
row represents a single data point consisting of n-dimensional features x<sub>1</sub> 
... x<sub>n</sub> and m-dimensional labels y<sub>1</sub> ... y<sub>m</sub>. For 
unsupervised learning tasks, no labels are available. For unregular datasets, n 
and m represent the dimension of the longest feature and label vectors available 
in our datasets; every data point/row then only contains features up to the relevant 
vector entry, and is sparse for the remaining parts.



### Getting started

Download this repository to your home directory:

```
cd 
git clone https://github.com/ArsamAryandoust/EnergyTransitionTasks
cd EnergyTransitionTasks
```

### Docker

Build and run Docker for processing data:

```
docker build -t main Docker
docker run -v ~/EnergyTransitionTasks:/EnergyTransitionTasks main
```

Build and run Docker for processing data with Jupyter notebooks:

```
docker build -t main_notebook DockerNotebook
docker run -it -v ~/EnergyTransitionTasks:/EnergyTransitionTasks -p 3333:1111 main_notebook
```

Open the link that shows in your terminal with a browser. Then, replace the port 
1111 with 3333 in your browser link to see notebooks inside the Docker container.


### Contributions

If you have published or unpublished datasets that fit our purpose, your contribution
is highly appreciated. For changes to code, please download this repository, create 
a new branch, push your changes to the repository, and create a pull request to 
the latest\_release branch.

If you are using the datasets we provide here and have identified any problems 
with these or want to suggest ways to improve these, your feedback is highly 
appreciated. For this, please go to the Discussions tab, create a New discussion,
and give us your detailed feedback.

---
In the following, we provide a short summary of each prediciton task and dataset, 
and its relevance and potential pitfalls for enhancing the global energy transition.

### Uber Movement

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


### Open Catalyst

The Open Catalyst project provides data on about 1.3 million molecular relaxations
with results from over 260 million density functional theory calculations from
quantum mechanical simulations under a CC Attribution 4.0 International licence
(https://creativecommons.org/licenses/by/4.0/legalcode). The goal is to predict 
and find cost-effective electrocatalysts for producing synthetic fuels like 
hydrogen with excess renewable energy from intermittent wind and and solar [2]. 
Common state-of-the-art electrocatalysts for both producing hydrogen and consuming
it again use expensive noble metals such as iridium and platinum. If these can be
replaced by new cost-effective catalysts, an earlier adoption of hydrogen energy
storage systems may be achieved due to significant cost reductions. Data from the
periodic table of elements is collected from here: https://pubchem.ncbi.nlm.nih.gov/periodic-table/.

A critical view on hydrogen is provided in https://blogs.ethz.ch/energy/h2-or-not/

### ClimART

The ClimART project provides data on the inputs and outputs of the Canadian Earth
System Model for emulating atmospheric radiative transfer between the years 1850 
and 2100 under a CC BY 4.0 license (https://creativecommons.org/licenses/by/4.0/).
The computation of atmospheric radiative transfer is the computational bottleneck 
for current weather and climate models due to its large complexity [3]. Training
machine learning models that are able to predict such model outputs at faster 
computational speed can therefore fill the information gaps for times and places
where earth system model are too computationaly expensive to be run.


### Building Electricity

The Building Electricity dataset provides electricity consumption profiles of about 
100 and 400 single buildings in 2014, aerial imagery data of these buildings and
and meteorlogical conditions in the area of these buildings during the same year
under an MIT license (https://opensource.org/licenses/MIT). The goal is to predict
the electric consumption profile of single buildings for the next 24h in 15-min
time steps (96 values) at different times as the only ground truth data in the 
prediction task from only remotely sensed features consisting of meterological data
and aerial imagery of buildings [4].


---

### References

[1] Aryandoust, A., Van Vliet, O. & Patt, A. City-scale car traffic and parking 
density maps from Uber Movement travel time data, Scientific Data 6, 158 (2019). 
https://doi.org/10.1038/s41597-019-0159-6

[2] Zitnick, C. L. & et al. An Introduction to Electrocatalyst Design using Machine 
Learning for Renewable Energy Storage. Preprint at arxiv (2020).
https://doi.org/10.48550/arXiv.2010.09435

[3] Cachay, S. R., Ramesh, V., Cole, J. N. S., Barker, H. & Rolnick D. ClimART:
A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and 
Climate Models. In Proc. of the 35th Conference on Neural Information Processing 
Systems Track on Datasets and Benchmarks (2021). https://doi.org/10.48550/arXiv.2111.14671

[4] Aryandoust, A., Patt, A. & Pfenninger, S. Enhanced spatio-temporal electric 
load forecasts using less data with active deep learning, Nature Machine 
Intelligence 4, 977-991 (2022). https://doi.org/10.1038/s42256-022-00552-x

