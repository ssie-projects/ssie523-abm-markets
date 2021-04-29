### Overview of Parameter Optimzation Approach.

- Downloaded ground-truth hourly data from Yahoo for SPY from 2006 to April 26, 2021. With these data, the log-returns was computed from the closing price, and day-over-day difference was computed. Once the log-returns were calculated, summary statistics for the dataset was calculated which included, mean, variance, square variance, skew, and kurtosis (collectively, "four moments"). The notebook to generate the moments can be downloaded [here.](https://github.com/ssie-projects/ssie523-abm-markets/blob/lux-abm/research/param-opt/ground-truth-stats.ipynb) The ground-truth statistics will be used in calculating the overall error when screening parameter settings for the model.

- A parameter space was developed by computing the carteisan product for four parameters (`nu`, `tf`, `tc`, `b`). The resulting parameer space containted 45,000,000 settings. The workbook that generated the parameter space is [here.](https://github.com/ssie-projects/ssie523-abm-markets/blob/lux-abm/research/param-opt/parameter-space-generation.ipynb)
	- nu \in `list(np.linspace(0.0001, 1, 10000))`
	- tf \in `list(np.linspace(1, 30, 30))`
	- tc \in `list(np.linspace(1, 30, 30))`
	- b \in `list(np.linspace(0.001, 0.999, 5))`

- Note that the pkl file that contains the parameter space is not uploaded to github since the filesize is > 100 mb; however, you can [download parameter space](https://drive.google.com/file/d/1KWnXbysBJ1Gd56JmDCjj2K7o5T2DTEY9/view?usp=sharing), as a `pkl` file that contains 45,000,000 parameter settings and add it to the directory.

- The idea to train the model against the raw data was to compute the percent difference between the ground-truth data and the model output for the four moments (percent difference calculated as `100 * (2 * (a - b) / (a + b))`). Once the percent difference for each of the moments was computed, a single error value was computed as the average of the absolute value of the percent difference for each moment. That is, for each moment, the percent difference was calculated, the absolute value calculated, and the average was computed using each of the moments. Even though the average is sensitive to extremes, I felt that this was a good initial measure to get an idea of the overall error. 

- To compute the error, I needed to create a new attribute in the class that starts on line 216 of this [code base.](https://github.com/ssie-projects/ssie523-abm-markets/blob/2f334837e3161c4d7e18a30e66b845375e623abf/research/param-opt/1LuxABM-gta-param-tuning.py#L216) The values seen on [line 232 ](https://github.com/ssie-projects/ssie523-abm-markets/blob/2f334837e3161c4d7e18a30e66b845375e623abf/research/param-opt/1LuxABM-gta-param-tuning.py#L232) are from the ground truth data for SPY.

- A sweep was performed on the first 50,000 parameter settings was performed and the results were logged. Using the current code, it takes about 4 min to run through 1,000 parameter sets. Given the time it takes to run, it's not going to be possible to run through the entire 45 MM parameters, and so I limit the sweeps to 50K at a time by truncating the size of the list as noted on [line 292.](https://github.com/ssie-projects/ssie523-abm-markets/blob/2f334837e3161c4d7e18a30e66b845375e623abf/research/param-opt/1LuxABM-gta-param-tuning.py#L292)

- The parameter set with the lowest average percent difference was selcted as the optimzed parameter set. From the parameter sweep, the lowest error was computed to be ~15.123% with the following: `{'tf': 1.0, 'tc': 28.0, 'nu': 0.0007000000000000001, 'b': 0.2505}}` (see [here](https://github.com/ssie-projects/ssie523-abm-markets/blob/2f334837e3161c4d7e18a30e66b845375e623abf/research/param-opt/candidates1.txt#L453))

- With the set of parameters from the initial sweep, I then proceeded to compute the moments using the [initial model that Dan built ](https://github.com/ssie-projects/ssie523-abm-markets/blob/lux-abm/research/param-opt/1LuxABM-gta-opt-param.ipynb)(which I understand is the basis for the class that was created by Christian).

- I also used the same set of parameters to get the moments using the [code base generated from Christian.](https://github.com/ssie-projects/ssie523-abm-markets/blob/lux-abm/research/param-opt/1LuxABM-gta-opt-param.py)

### Problems 

- It seems that the returns that are generated from Dan's code don't look normal. It seems that the model is chaotic; however, I have not looked into this in great detail. (see [here](https://github.com/ssie-projects/ssie523-abm-markets/blob/lux-abm/research/param-opt/1LuxABM-gta-opt-param.ipynb))

- Between both the model output from Dan and Christian versions, it seems that the moments are different (I left the other values).

Christian:
kurt: 27.780
skew: -0.621
mu: 0.000293

Dan:
kurt: 12.656
skew: -1.373
mu: 0.000378

- One possible fix is to seed the code base that is used to do the parameter sweep. 

- At this point, I'd like to discuss the approach for testing each parameter setting. Does the approach make sense? What changes can we make? Once I get feedback, I can sample through the parameter space and search for another set or create a new parameter set using a log scale. 

There is a lot of detail that is better discussed. 
