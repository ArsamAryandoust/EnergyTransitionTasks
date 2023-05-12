# Uber Movement

n := number of data points <br />

| Variable | Description | Format |
| --- | ----------- | ----------- |
| x_t | time-variant features: year, quarter of year, hour of day, day type | (n, 4) |
| x_s_1 | space-variant features 1: city ID| (n, 1) |
| x_s_2 | space-variant features 2: 3D geographic Cartesian coordinates (x, y, z) of centroids on a unit sphere for origin-destination pair of city zone polygons. | (n, 6) |
| y_st | labels: mean, std deviation, geographic mean, geographic std deviation of travel time between given origin-destination pair of zones at given time | (n, 4) |

