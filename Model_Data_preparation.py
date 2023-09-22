#Script to prepare the data for model input
#import dependencies
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
#%matplotlib qt
import os
import holidays


def weather_day_merge(zone_name, config_data):
    '''
    function to merge zonal elec demand data with the weather data,i.e. feels like temperature and calendar information
    config_data = config dictionary
    output: Merged dataframe consisting of demands, weather info and holiday/weekend info
    '''
    zone_name = zone_name
    zonal_data_path = config_data['zonal_data_path'] #get the path of zonal demand data
    weather_data_path = config_data['weather_data_path'] #get the path of weather data
    start_date = config_data['start_date'] #get the start date of extracting the data
    end_date = config_data['end_date'] #get the end date of extracting the data
    df_demand_en = pd.read_csv(zonal_data_path+zone_name+'_demand.csv') #read the demand data as a dataframe
    df_demand_en['Datetime'] = pd.to_datetime(df_demand_en['Datetime']) 
    df_demand_en = df_demand_en.set_index('Datetime') #set index to datetime to perform merge operation with other dataframes like weather
    
    #start_date = '2015-06-30 23:00:00'
    #end_date = '2018-10-01 00:00:00'
    # Select DataFrame rows between two dates
    mask = (df_demand_en.index > start_date) & (df_demand_en.index <= end_date)
    df_demand_en = df_demand_en.loc[mask]
    #filter out a subpart of the dataframe to be used in training and testing, loc() used to select row/column
    
    
    df_weather = pd.read_csv(weather_data_path) #read weather data
    # df_weather['datetime'] = pd.to_datetime(df_weather['dt'],unit='s')
    df_weather.index= pd.to_datetime(df_weather['dt_iso'])
    # df_demand_en.index = pd.to_datetime(df_demand_en.index).tz_localize(None) #Converting to same timestamp
    # df_weather.index = pd.to_datetime(df_weather.index).tz_localize(None)


    # weather_data = df_weather.drop(['level_0','index','feelslike'], axis = 1) #drop all the info apart from temperature data
    temperature_data = pd.DataFrame(df_weather['feelslike'],columns = ['feelslike'])
    # weather_temp = weather_data.drop(['dt_iso'], axis=1)
    # weather_temp.drop_duplicates(keep=False,inplace=True) #drop duplicate entries if any, to avoid problems with merging
    # weather_temp = weather_temp.set_index('datetime')

   

   

    df_demand_en = pd.merge(df_demand_en, temperature_data,  left_index=True, right_index=True) 
    #join the table of demand and temperature data
   
    df_demand_en = df_demand_en.rename(columns = {"em":"cons"})
    # df_demand_en = df_demand_en.dropna() #droping the first row with NaN generated for differencing

    df_demand_en = df_demand_en.reset_index() #index turns to new column "index"

    df_demand_en["DayOfWeek"] = df_demand_en["index"].dt.weekday 
    #extract day of week from the dates,dt: access datetime measures, weekday gives integer (0monday to 6 sunday)
    
    dayofweek = df_demand_en['DayOfWeek'].values 
    #day of week is from 0 to 6 for Monday to Sunday. For saturday or sunday, day of week is binarized as 1, and 0 for others below

    for ii in range(len(dayofweek)):
        if dayofweek[ii]>4:
            dayofweek[ii]=1
        else:
            dayofweek[ii]=0


    df_demand_en['HolidayOrWeekend'] = dayofweek 
    #create a separate column for combining holiday info with weekend as we are considering holidays to have same identity as weekends
    
    df_demand_en['Date'] = df_demand_en['index'].dt.date
    lu_holidays = holidays.LU() #get the list of holidays for Luxembourgh
    # print("holidays:",lu_holidays)
    holiday = np.zeros(len(df_demand_en))
    for ii in range(len(df_demand_en['Date'].values)):
        Date = df_demand_en['Date'].values[ii] 
        isHoliday = lu_holidays.get(Date) #checking which of the dates in our dataframe are there in holdiay-list dates
        if isHoliday!=None:
            holiday[ii] = 1
    df_demand_en['HolidayOrWeekend'] = df_demand_en['HolidayOrWeekend'].values + holiday #combining weekends with holidays
    df_demand_en["HolidayOrWeekend"] = np.where(df_demand_en["HolidayOrWeekend"] > 0, 1, 0) #binary encoding of dates to 0 and 1
    return df_demand_en

def _generate_time_frequency(time_series, wavelet, num_freq):
    '''
    Function to generate scalograms for a time series
    inputs: time_series = time series data within the analysis window
    wavelet = Basis function of mother wavelet
    num_freq = number of scales for scalogram generation
    Output:
    returns the coefficients of CWT in the form of wavelet scalogram
    '''
    wavelet_function = wavelet
    num_frequencies = num_freq
    wavelet_transformed, freqs = pywt.cwt(time_series, range(1, num_frequencies + 1), wavelet_function)
    return wavelet_transformed


def model_input_prep(zone_name, config_data):
    '''
    function to prepare the data for train test split
    input: zone_name = zone name, 
    config_data = config data dictionary
    output:
    XWH = combined data matrix of scalograms of elec consumption of current day, weather data for current and next day, holiday/Weekend/Weekday binary representation of current and next day
    y = target sequences for training /validation/testing
    df_demand_en = outlier removed dataframe combined with endogeneous and exogenous information
    min_elec = minimum value of demand for inverse scalarization
    max_elec = maximum value of demand for inverse scalarization
    '''
    df_demand_en = weather_day_merge(zone_name, config_data) #get the zonal elec demand data merged with weather and calendar info
    mu = df_demand_en['cons'].mean() #get the mean of elec demand
    #print(mu)
    sigma = df_demand_en['cons'].std() #get the standard deviation of elec demand
    #print(sigma)
    Train_Days = config_data['Train_Days'] #number of days for training
    val_days = config_data['val_days'] #number of days for validation
    Test_Days = config_data['Test_Days'] #number of days for testing
    pred_window=config_data['pred_window'] #size of forecasting window, e.g. 24 hours
    hours_lookback=pred_window
    num_freq = config_data['num_frequencies'] #number of scales
    

    days = Train_Days + val_days + Test_Days #total days to prepare the dataset for train-test-validation
    print("days:",days)
    
    
    

    X = np.zeros((days,hours_lookback)) #initializing the array of size: days x hours to look back,e.g. 730x24, where each row vector corresponds to sequence of 24 hours observations of current day
    print("x:",X.shape)
    y = np.zeros((days,hours_lookback))
    print("y:",y.shape)
    #initializing the array of size: days x prediction window (as prediction window=hours to look back),e.g. 730x24, where each row vector corresponds to sequence of 24 hours observations of next day, as target sequence
    W = np.zeros((days,hours_lookback)) #initializing the array of size: days x hours to look back ,e.g. 730x24, where each row vector corresponds to sequence of 24 hours observations of feel like temperature of current day
    Wf = np.zeros((days,hours_lookback))
#initializing the array of size: days x hours to predict ,e.g. 730x24, where each row vector corresponds to sequence of 24 hours weather forecasts  of feel like temperature 
    ##incorporating holiday
    H = np.zeros((days,hours_lookback)) #initializing the array of size: days x hours to predict ,e.g. 730x24, where each row vector corresponds to sequence of zeros or ones, depending on whether the current day is a weekday or holiday/weekend
    Hf = np.zeros((days,hours_lookback))
#initializing the array of size: days x hours to predict ,e.g. 730x24, where each row vector corresponds to sequence of zeros or ones, depending on whether the next day is a weekday or holiday/weekend
    holiday = df_demand_en['HolidayOrWeekend'].values

    elec = df_demand_en['cons'].values
    print(elec)
    print(elec.shape)
    #remove the observations belonging to statistical outliers of elec demand, possibily generated as measurement errors
    for ii in range(len(elec)):
        if elec[ii]>2*(mu+sigma):
            elec[ii] = elec[ii-1]
    
    min_elec = elec.min()
    max_elec = elec.max()
    
    elec = (elec-elec.min())/(elec.max()-elec.min()) #Min-max scalerize elec demand
    print(elec)
    print(elec.shape)

    weather = df_demand_en['feelslike'].values
    weather = (weather-weather.min())/(weather.max()-weather.min()) #minmax scalerize feel like temperature

  

    for ii in range(days):
        #making chunk of sequences of all the inputs and target
      
        # print("ii+1:", ii+1, "hours loopback:", hours_lookback, "y len:", (ii+1)*hours_lookback - (ii+1)*hours_lookback+hours_lookback)
        X[ii] = elec[ii*hours_lookback:ii*hours_lookback+hours_lookback]
        y[ii] = elec[(ii+1)*hours_lookback:(ii+1)*hours_lookback+hours_lookback]
        W[ii] = weather[(ii)*hours_lookback:(ii)*hours_lookback+hours_lookback]
        Wf[ii] = weather[(ii+1)*hours_lookback:(ii+1)*hours_lookback+hours_lookback]
        H[ii] = holiday[(ii)*hours_lookback:(ii)*hours_lookback+hours_lookback]
        Hf[ii] = holiday[(ii+1)*hours_lookback:(ii+1)*hours_lookback+hours_lookback]

        '''
        if ii<Train_Days:
            W[ii] = weather[(ii)*24:(ii)*24+24]
            H[ii] = holiday[(ii)*24:(ii)*24+24]
        if ii>Train_Days:
            W[ii] = weather[(ii+1)*24:(ii+1)*24+24]
            H[ii] = holiday[(ii+1)*24:(ii+1)*24+24]
        '''
#Initializing the scalogram matrix/binary encoding matrix with the size of (days,scales, hours to lookback,1)
#e.g. (730,24,24,1) for the scalograms of 1 input variable
#the 1 at the end is for appending them like channels of image
#e.g. (730,24,24,5) for 5 channel input
#Each day's scalogram shape: (24,24,1)
    
    print("x:",X.shape)
    print("y:",y.shape)
   
    XX = np.zeros((days,num_freq,hours_lookback,1))
    WW = np.zeros((days,num_freq,hours_lookback,1))
    WF = np.zeros((days,num_freq,hours_lookback,1))
    HH = np.zeros((days,num_freq,hours_lookback,1))
    HF = np.zeros((days,num_freq,hours_lookback,1))
#Generating scalograms of current day's elec consumption
    for ii in range(days):
        wav  = _generate_time_frequency(X[ii], config_data['wavelet_function'], num_freq)
        wav = wav.reshape((num_freq,hours_lookback,1))
        XX[ii] = wav
#Generating scalograms of current day's feel like temperature
    for ii in range(days):
        wav  = _generate_time_frequency(W[ii] , config_data['wavelet_function'], num_freq)
        wav = wav.reshape((num_freq,hours_lookback,1))
        WW[ii] = wav
#Generating scalograms of next day's feel like temperature, i.e. weather forecast
    for ii in range(days):
        wav  = _generate_time_frequency(Wf[ii], config_data['wavelet_function'], num_freq)
        wav = wav.reshape((num_freq,hours_lookback,1))
        WF[ii] = wav
#Creating a matrix of all zeros or all ones depending whether the current day is a weekday or weekend/holiday
    for ii in range(days):
        if np.sum(H[ii])!=0:
            HH[ii] = np.ones((num_freq,hours_lookback,1))
#Creating a matrix of all zeros or all ones depending whether the next day is a weekday or weekend/holiday

    for ii in range(days):
        if np.sum(Hf[ii])!=0:
            HF[ii] = np.ones((num_freq,hours_lookback,1))

    XWH = np.concatenate((XX,WW,WF, HH, HF), axis=3) #Append the scalograms and binary matrices by channels
    # XWH = np.concatenate((XX,HH, HF), axis=3) #3channel
    #XWH = np.concatenate((XX,WW), axis=3)
    print("xwh:",XWH.shape)
    print("XX:",XX.shape)
    # print(XWH)
    return XWH,y,df_demand_en,min_elec, max_elec


def train_test_split(XWH, y, val_days, Test_Days):
    '''
    function to split the training, validation and test data out of the data matrix
    input: XWH = Data matrix, i.e. scalograms and binary encodings combined by channels
    y=target sequences
    val_days = validation days
    Test_Days = testing days
    '''

    train_split = val_days+Test_Days

    X_train  = XWH[:-train_split] #Get the training features
    X_test = XWH[-train_split:] #Get the testingresult_df feature

    y_train = y[:-train_split] #Get the training target sequences
    y_test = y[-train_split:] #Get the testing+validation sequences

    y_val = y_test[:-val_days] #Get the validation output sequences
    X_val = X_test[:-val_days] #Get the validation input sequences

    X_test = X_test[-Test_Days:] #Get the testing input sequences
    y_test = y_test[-Test_Days:]  #Get the testing output sequences


   
    print("y-test:",y_test.shape)



    return X_train, y_train, X_val, y_val, X_test, y_test


