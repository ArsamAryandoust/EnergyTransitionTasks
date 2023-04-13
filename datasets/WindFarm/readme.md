# Wind Farm


| Feature | Description | Format |
| --- | ----------- | ----------- |
| x_t_1 | 288 time stamps of historic values containing day, hour, 10-min | (n, 3, 288) |
| x_t_2 | 288 time stamps of requested prediction window containing day, hour, 10-min | (n, 3, 288) |
| x_s | x and y Cartesian positions of turbines in meters | (n, 2) |
| x_st | Turbine state data for every past time stamp consisting of wind speed recorded by the anemometer in m/s, angle between the wind direction and the position of turbine nacelle in °, temperature of the surrounding environment in °C, temperature inside the turbine nacelle in °C, the yaw angle of the nacelle in °, the pitch angles of all three blades in °, the reactive power generation in kW and the active power generation of the turbine in kW | (n, 10, 288) |


| Label | Description | Format |
| --- | ----------- | ----------- |
| y | 24h of 15-min future electric load profile of given building at given time | (n, 96) |


| Additional data | Description | Format |
| --- | ----------- | ----------- |
| id_histo_map | A mapping of building IDs to RGB histogram data of a buildings aerial image. This can be used to dynamically, or statically, expand x_s into the recommended format of (n, 300) or (n, 100, 3). Each subtask has a different file with 92 and 451 buildings respectively. | (300, 92) (300, 451) |

n := number of data points <br />
x_t := time-variant features <br />
x_s := space-variant features <br />
x_st := space-time-variant features <br />
y := labels

