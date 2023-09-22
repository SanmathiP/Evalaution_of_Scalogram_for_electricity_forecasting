import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd
from datetime import datetime
current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

def make_prediction(zone_name, zonal_merged_df, model, history, X_test, y_test,  min_elec, max_elec,config_data):
    '''function to make predictions for all the testing days
    inputs: zone_name = name of the zone for which prediction is taking place
    zonal_merged_df = Merged dataframe of zonal demand, weather and calendar information
    model = Trained model
    X_test = Testing inputs
    y_test = Ground truths for testing
    min_elec = minimum elec demand for inverse scalaring the normalized elec demand
    max_elec = maximum elec demand for inverse scalaring the normalized elec demand
    config_data = config data dictionary

    Output:
    dataframe including ground truth and prediction for all seasons in a dataframe for a particular zone
    '''
    y_pred = model.predict(X_test) #prediction using test inputs
    Y = np.abs(y_test)#.flatten()
    Y_hat = np.abs(y_pred) #just to ensure there is no negative predicted consumption, in case
    true_demand = (Y.flatten()*(max_elec-min_elec)) +  min_elec #inverse scalaring the ground truth
    pred_demand = (Y_hat.flatten()*(max_elec-min_elec)) +  min_elec #inverse scalaring predictions
    
    df_demand_en = zonal_merged_df
    Train_Days  = config_data['Train_Days']
    val_days = config_data['val_days']
    Test_Days =config_data['Test_Days']
    hours_lookback = config_data['hours_lookback']
    result_df = df_demand_en.head((Train_Days+val_days+Test_Days)*hours_lookback).tail(Test_Days*hours_lookback)
    # result_df = df_demand_en.tail((Train_Days+val_days+Test_Days)*hours_lookback).tail(Test_Days*hours_lookback)

    # print("result_df from evalute:",result_df)
    
    result_df['cons_pred'] = pred_demand #predicted demand in the result dataframe
    result_df['cons'] = true_demand #true demand in the result dataframe
    
    
    result_df.to_csv('/content/drive/MyDrive/proj_elec/Proj2/WavScaloNet_elec/results_csv/WavScaloNet_elec'+ zone_name+'_'+current_datetime+'_netresult_wi2so.csv')
    #save the results as csv export
    return result_df

def get_metrics(result_df, test_days, pred_window):
    '''
    function to evaluate the performance quantitatively
    input: result_df = result dataframe
    test_days = days of testing
    pred_window = forecasting window

    output:
    Dataframe containing performance metrics MAPE and RMSE for 1 zone
    '''
    mape_pred_demand= np.zeros(test_days)
   
    rmse_pred_demand= np.zeros(test_days)
    for ii in range(test_days):
        mape_pred_demand[ii] = 100*mape(result_df.cons.values[ii*pred_window:ii*pred_window+pred_window], result_df.cons_pred.values[ii*pred_window:ii*pred_window+pred_window]) #computing accuracy = (1-MAPE)*100%
        rmse_pred_demand[ii] = np.sqrt(mse(result_df.cons.values[ii*pred_window:ii*pred_window+pred_window], result_df.cons_pred.values[ii*pred_window:ii*pred_window+pred_window])) #computing RMSE
    # print('mape:',mape_pred_demand)
    # print('rmse:',rmse_pred_demand)
    metrics_df = pd.DataFrame(mape_pred_demand, columns = ['pred_demand_acc'])
    metrics_df['pred_demand_rmse'] = rmse_pred_demand
    metrics_df['pred_demand_mape'] = mape_pred_demand  
    metrics_df.to_csv('/content/drive/MyDrive/proj_elec/Proj2/WavScaloNet_elec/error_metrics_outputs/'+'err_'+current_datetime+'.csv')

    # # Calculate accuracy and precision
    # accuracy = accuracy_score(result_df.cons, result_df.cons_pred)
    # precision = precision_score(result_df.cons, result_df.cons_pred)
    # metrics_df['accuracy'] = accuracy
    # metrics_df['precision'] = precision
    return metrics_df