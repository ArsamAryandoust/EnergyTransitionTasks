# Uber Movement

Aryandoust et al. use travel time changes between different zones of a city to infer probabilities of driving and choosing a destination, and sample spatio-temporal car parking maps on the scale of entire cities with these [1]. Their car parking maps are primarily used for planning the distribution and sizing of charging stations in renewable power systems when electrifying mobility and utilizing storage capacities for vehicle-to-grid electricity system services. We can predict origin-destination matrices of travel time between any two city zones at a given time, given the geographic information of city zones and information about circadian rhythms of a city. This allows us to expand the analysis of car parking maps to cities where no travel time data is available, or improve parking maps for cities where data is missing and travel time matrices are highly sparse. Missing data itself, however, contains information like a lack of traveling demand between two city zones at a given time. Predicting missing travel time data therefore bears the risks of worsening the performance of the downstream task for sampling parking maps by filling informative sparsity.

The authors use travel time data published by the Uber Movement project (\url{https://movement.uber.com}) that are currently available under a CC BY-NC 3.0 US license (\url{https://creativecommons.org/licenses/by-nc/3.0/us/}) for 51 cities around the globe. The geographic information of city zones is further available under a custom license for each city.  


### Citation:
[1] Aryandoust, A., Van Vliet, O. & Patt, A. City-scale car traffic and parking 
density maps from Uber Movement travel time data, Scientific Data 6, 158 (2019). 
https://doi.org/10.1038/s41597-019-0159-6
