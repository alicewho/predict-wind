This project uses One-Step Ahead Linear Regression to predict wind direction one timestamp/step in the future. 

The data consists of real IoT weather data collected in Aarhus Denmark and all text files are available here: http://iot.ee.surrey.ac.uk:8080/datasets.html#weather

Readings are collected at variable time intervals; this project assumed equal time intervals. Interpolation could be used for further improvements. 

Because wind direction is given initially in angle values, it is a bounded quantity from 0-360, so continuous wind vector x- and y-components were used instead. 

Elastic Net performs well and results in a small mse in both cross-validation and test. 

Also included are several plots for visualizing the results as well as post-processing to get back actual wind direction, etc. 


