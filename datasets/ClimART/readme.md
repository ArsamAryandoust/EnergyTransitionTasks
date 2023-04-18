# ClimART

n := number of data points <br />

| Variable | Description | Format |
| --- | ----------- | ----------- |
| x_t | time-variant features: year and hour of year in 205h steps| (n, 2) |
| x_s | space-variant features: 3D geographic Cartesian coordinates (x, y, z) on unit sphere | (n, 3) |
| x_st_1 | space-time-variant features 1: global variables. Originally been 82, but now 79 as we exclude 3D geographic coordinates that are now in x_s | (n, 79) |
| x_st_2 | space-time-variant features 2: 14 (pristine) or 45 (clear_sky) variables for every layer (total 49) | (n, 49, 14)/(n, 49, 45) |
| x_st_3 | space-time-variant features 3: 4 variables for every level (total 50)| (n, 50, 4) |
| y_st_1 | labels 1: solar and thermal heating rate profiles for each layer in K/s | (n, 49, 2) |
| y_st_2 | labels 2: Up- and down-welling shortwave solar and longwave thermal flux for each level in W/m² | (n, 50, 4) |

