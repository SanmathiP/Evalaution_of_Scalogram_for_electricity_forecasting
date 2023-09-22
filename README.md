# Electricity_prediciton
Evaluating the Wavelet Scalogram Framework for Electricity Forecasting

The Deep Learning Framework of forecasting which has the combination of the Wavelet transform method covers the frequency and temporal aspects.
Training through the neural network helps it deal with complex and non linear nature of the consumption data, CNN in particular due to image-like representation of scalograms. 
The method considers effects from exogenous variables like Weather and Information of Type of Day(holiday/non holiday)
Having additional variables are also flexible on including and extending to various other exogenous variables.
Here, the framework forecasts for 24 steps, but it can be further used to forecast for over 24 steps/hours to multiple days.

# Methodology
(1)A Scalogram representation based framework capturing localized non-linear time-frequency features of electricity
consumption with and without all exogenous variables;
(2) A 24 steps ahead electricity consumption forecast by a deep learning model; 
(3) Forecast in various seasons and their performance evaluation with standard deep learning method.
![methodology2](https://github.com/SanmathiP/Electricity_prediciton/assets/75175133/528ef1d7-7828-4d55-a6a9-33b4a004257d)

All the three experiments are compared with the LSTM model. The experiments are run for four
seasons of the year.
The 5 channel outputs are more accurate when compared
to 3 channel output, the 3 channel outputs under-forecast slightly..
![image](https://github.com/SanmathiP/Electricity_prediciton/assets/75175133/d528afbd-513b-4797-9f55-04023aeca27e)
![image](https://github.com/SanmathiP/Electricity_prediciton/assets/75175133/9855ba5b-5242-4573-9c11-bb4153c97175)
![spring_15mins](https://github.com/SanmathiP/Electricity_prediciton/assets/75175133/00b7b74c-b8d9-43eb-bf35-d1a0c6bf10c4)
Though the quantitative results MAPE and RMSE are are almost same for the proposed model and LSTM, with 15 minute data has the least MAPE and RMSE values, whereas 3 channeled data as expected has high error values![mape2](https://github.com/SanmathiP/Electricity_prediciton/assets/75175133/aaec780d-f244-4a65-9d3b-09bfd55d24e5)
![rmse2](https://github.com/SanmathiP/Electricity_prediciton/assets/75175133/7b08af3b-4635-4243-a728-fdb6203125b8)
