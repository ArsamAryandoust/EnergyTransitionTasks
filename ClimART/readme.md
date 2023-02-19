# ClimART

In a project called ClimART, Cachay et al. predict the athmospheric radiative transfer using meteorological data for the chemistry at and between each layer of the atmosphere from the Canadian Earth System Model [1]. Although the functional relationships used in climate and weather models are based on well-understood physics, values like the atmospheric radiative transfer can be complex to calcualte and create a computational bottleneck for modeling weather and climate at high spatio-temporal resolutions. AI and ML models can emulate these relationships merely based on the inputs and outputs of these models and have the advantage of being significantly faster at inference time as compared to the simulation time of the underlying physics-informed models. This allows us to model weather and climate at much higher resolutions, and update our calculations more frequently. Improved climate projections allow us to better understand weather regimes such that we reduce the demand for storage and generation capacities, achieve energy autarky in regions where it is necessary and guarantee the security of supply when planning transmission grids in large renewable electricity systems.

The authors provide data on the inputs and outputs of the climate model between the years 1850 and 2100 under a CC BY 4.0 license (\url{https://creativecommons.org/licenses/by/4.0/}) on a self-hosted remote repository that can be accessed with instructions provided on Github under \url{https://github.com/RolnickLab/climart}. Their prediction task includes two datasets, which have the same features for each data point and only differ in labels, that is, the atmospheric radiative transfer we want to predict: in the first task, which we refer to as the \emph{pristine-sky} dataset, no aerosol and no clouds are considered which makes this task simpler; in the second task, which we refer to as the \emph{clear-sky} dataset, the impact of aerosol and clouds on radiative transfer is further considered leading to more values to predict.


### Citation:
[1] Cachay, S. R., Ramesh, V., Cole, J. N. S., Barker, H. & Rolnick D. ClimART:
A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and 
Climate Models. In Proc. of the 35th Conference on Neural Information Processing 
Systems Track on Datasets and Benchmarks (2021). https://doi.org/10.48550/arXiv.2111.14671
