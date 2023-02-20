# Building Electricity

Aryandoust et al. predict the electric load profile of single buildings for a future time window of 24 h at a given time, given purely remotely sensed features, that is, the aerial image of buildings and the meteorological conditions in the region of these buildings for a past time window of 24 h [1]. Among many different types of electric load forecasts that are useful for enhancing the energy transition, this falls into the category of short-term spatio-temporal forecasts and can be used for both planning and dispatch of renewable systems. The task aims at assessing the information content of candidate data points prior to physically measuring or streaming these, a method known as active learning, so as to reduce the use of smart meters and data queries to the most informative data only. Enhancing solutions to this task at scale, however, bears the risk of having significantly larger computation-related GHG emissions compared with distributing smart meters and querying their data at random using passive learning, as the assessment of informativeness of data points is found to cost 4-11 times more computational time in the original study.

The authors publish their datasets under an MIT license (https://opensource.org/licenses/MIT) in a Harvard Dataverse repository (https://doi.org/10.7910/DVN/3VYYET). Their prediction task includes two datasets, one containing 92 buildings which we refer to as the building-92 dataset, and the other containing 459 buildings which we refer to as the building-459 dataset.

### Citation:
[1] Aryandoust, A., Patt, A. & Pfenninger, S. Enhanced spatio-temporal electric 
load forecasts using less data with active deep learning, Nature Machine 
Intelligence 4, 977-991 (2022). https://doi.org/10.1038/s42256-022-00552-x

