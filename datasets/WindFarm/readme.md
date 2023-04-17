# Wind Farm

n := number of data points <br />

| Variable | Description | Format |
| --- | ----------- | ----------- |
| x_t_1 | time-variant features 1: 288 time stamps of historic values containing day, hour, 10-min | (n, 3, 288) |
| x_t_2 | time-variant features 2: 288 time stamps of requested prediction window containing day, hour, 10-min | (n, 3, 288) |
| x_s | space-variant features: x and y Cartesian positions of turbines in meters | (n, 2) |
| x_st | space-time-variant features: Turbine state data for every past time stamp consisting of wind speed recorded by the anemometer in m/s, angle between the wind direction and the position of turbine nacelle in °, temperature of the surrounding environment in °C, temperature inside the turbine nacelle in °C, the yaw angle of the nacelle in °, the pitch angles of all three blades in °, the reactive power generation in kW and the active power generation of the turbine in kW | (n, 10, 288) |
| y_st | labels: 288 of electric power generation of turbine for each requested future time stamp | (n, 288) |


