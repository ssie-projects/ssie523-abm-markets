### Parameter Optimzation
[Download Paramter Space](https://drive.google.com/file/d/1KWnXbysBJ1Gd56JmDCjj2K7o5T2DTEY9/view?usp=sharing), as a `pkl` file that contains 45,000,000 parameter settings.

### Overview of Parameter Optimzation Approach.

- Downloaded ground-truth hourly data from Yahoo for SPY from 2006 to April 26, 2021. With these data, the log-returns was computed from the closing price, and day-over-day difference was computed. Once the log-returns were calculated, summary statistics for the dataset was calculated which included, mean, variance, square variance, skew, and kurtosis (collectively, "four moments").

- A parameter space was developed by computing the carteisan product for the four parameters. The resulting parameer space containted 45,000,000 settings. 

- The initial model was modified to compute the percent difference (calculated as `100 * (2 * (a - b) / (a + b))`) for each of the moments. That is, the percent difference between the respective moment from the ground-truth data and the model-generated data was computed. The average of the absolute value of the percent difference for each moment was computed and used as the final error for each parameter. 

- A sweep was performed on the first 50,000 parameter settngs was performed and the results logged. 

- The parameter set with the lowest average percent difference was selcted as the optimzed parameter set. From the parameter sweep, the lowest error was computed to be ~15.123% with the following: `{'tf': 1.0, 'tc': 28.0, 'nu': 0.0007000000000000001, 'b': 0.2505}}`