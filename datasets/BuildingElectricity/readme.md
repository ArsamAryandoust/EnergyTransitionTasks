# Building Electricity

n := number of data points <br />

| Variable | Description | Format |
| --- | ----------- | ----------- |
| x_t | time-variant features: time stamp containing year, month, day, hour, quarter of hour | (n, 5) |
| x_s | space-variant features: building ID that can be mapped to RGB image pixel histograms | (n, 1) |
| x_st | space-time-variant features: 24h of hourly past meteorological data for air density in kg/m³, cloud cover, precipitation in mm/hour, ground level solar irradiance in W/m², top of atmosphere solar irradiance W/m², air temperature in °C, snowfall in mm/hour, snow mass in kg/m² and wind speed  | (n, 9, 24) |
| y_st | labels: 24h of 15-min future electric load profile of given building at given time | (n, 96) |


| Additional data | Description | Format |
| --- | ----------- | ----------- |
| x_s | A mapping of building IDs (92 or 451 buildings) to RGB histogram data of a buildings aerial image. This can be used to dynamically, or statically, expand x_s into the recommended format of (n, 300) or (n, 100, 3). Each subtask has a different file with 92 and 451 buildings respectively. | (92, 100, 3) (451, 100, 3) |




