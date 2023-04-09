# Building Electricity

| Feature | Description | Format |
| --- | ----------- | ----------- |
| x_t | time stamp containing year, month, day, hour, quarter of hour | (n, 5) |
| x_s | building ID that can be mapped to RGB image pixel histograms | (n, 1) |
| x_st | 24h of hourly past meteorological data for air density, cloud cover, precipitation, radiation surface, radiation top of athmosphere, snow mass, snowfall, temperature  | (n, 24, 9) |


| Label | Description | Format |
| --- | ----------- | ----------- |
| y | 24h of 15-min future electric load profile of given building at given time | (n, 96) |


| Additional data | Description | Format |
| --- | ----------- | ----------- |
| id_histo_map | A mapping of building IDs to RGB histogram data of a buildings aerial image. This can be used to dynamically, or statically, expand x_s into the recommended format of (n, 300) or (n, 100, 3). Each subtask has a different file with 92 and 451 buildings respectively. | (300, 92) (300, 451) |

n:= number of data points <br />
x_t := time-variant features <br />
x_s := space-variant features <br />
x_st := space-time-variant features <br />
y := labels

Aryandoust et al. predict the electric load profile of single buildings for a future time window of 24h in 15-min steps, given purely remotely sensed features consisting of the aerial image of a building and the meteorological conditions in the region of a building for a past time window of 24h in 1-h steps [1]. Among many different types of electric load forecasts that are useful for enhancing the energy transition, this falls into the category of short-term spatio-temporal load forecasts, and can be used for both planning and dispatch of renewable electricity systems. The primary goal is to assess the information content of candidate data points prior to physically measuring or streaming these, a method known as active learning, so as to reduce the need for smart meters and data queries to the most informative data only. Enhancing solutions to this task, however, bears the risk of having significantly larger computation-related GHG emissions compared with distributing smart meters and querying their data at random using passive learning. This is because the assessment of informativeness of data points is found to cost 4-11 times more computation in the original study.

The authors publish their datasets under an MIT license (\url{https://opensource.org/licenses/MIT}) in a Harvard Dataverse repository (\url{https://doi.org/10.7910/DVN/3VYYET}). In order to anonymize the data, the authors provide only histograms of pixel values of aerial building images. The authors publish two sub-task datasets, one containing 92 buildings which we refer to as the \emph{buildings-92} dataset, and the other containing 451 buildings which we refer to as the \emph{buildings-451} dataset. Meteorological data are provided in time series with 15-min resolution so as match the electric load data resolution when splitting datasets, but is downscaled to the original 1-h resolution for the actual prediction task. Further detail on the datasets are provided in the original study [1].

### Citation:
[1] Aryandoust, A., Patt, A. & Pfenninger, S. Enhanced spatio-temporal electric 
load forecasts using less data with active deep learning, Nature Machine 
Intelligence 4, 977-991 (2022). https://doi.org/10.1038/s42256-022-00552-x

