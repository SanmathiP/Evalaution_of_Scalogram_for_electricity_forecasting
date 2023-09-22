#This script carries the CNN model structure and its training procedure
#import the necessary libraries
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
#%matplotlib qt
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler #for saving models
from keras.models import load_model

#imports for Randomsearch
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from tensorflow.keras.callbacks import ReduceLROnPlateau





def model_structure(num_freq, hours_lookback, num_channel,learning_rate=0.001):
    '''
    Function to initialize the model
    Function inputs: num_freq=Scales for scalogram
    hours_lookback = time window to look back, for e.g. previous 24 hours to forecast for next 24 hours
    num_channel=total number of inputs including endogenous and exogenous variables, e.g. 5 in case of current day's elec consumption,
    current day's weather, weather forecast, calendar information of current day and next day

    Funtion output: Initialized model
    '''
    #final model***********************NO POOLING model to avoid information loss
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(num_freq, hours_lookback, num_channel)))

    
    model.add(tf.keras.layers.Conv2D(32, (3,3), strides=1, padding='same', activation='relu'))
    #Conv2D layer with output size=32. kernel size=3,3, 'same' padding with 'relu' activation
    #same configuration type continues for further layers

    model.add(tf.keras.layers.Conv2D(64,(3,3),  padding='same', activation='relu'))
    
    model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    #flatten layer to get all the extracted features
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(2048, activation='leaky_relu')) #Fully connected layer with 2048 hidden units and leaky relu activation
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(168, activation='leaky_relu')) #Fully connected layer with 168 hidden units and leaky relu activation
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))                       #Drop out 20% to avoid overfitting
    #model.add(tf.keras.layers.Dense(64, activation='leaky_relu'))

    model.add(tf.keras.layers.Dense(24, activation='leaky_relu')) #Output layer to yield forecasts for 24 hours

    ############# for the sake of randomsearch ########
    # model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')


    return model

def train_model(model, X_train, y_train, X_val, y_val, epoch, batch_size):
# def train_model(model, X_train, y_train, X_val, y_val): for random search
    '''
    Function to train the model
    Inputs: model = initialized model
    X_train = Training features
    y_train=Training targets
    X_val=Validation features
    y_val= validation targets
    epoch= Maximum epochs for training
    Batch size= batch size for training

    Output: model = Trained model 
    history = the history of training and validation losses 
    '''
    # learning_rate = 0.001  # Set the learning rate
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate= 1e-3,
    #     decay_steps= 10,
    #     decay_rate= 0.8
    # )


   
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='MSE') #compilation of the model with adam optimizer and MSE loss function
    
    
    filepath='/content/drive/MyDrive/proj_elec/Proj2/WavScaloNet_elec/my_best_model.hdf5' #define the file path to save the models
    # filepath='my_best_model.hdf5'
    early_stop = EarlyStopping(monitor='val_loss', patience=50)#earlystopping to avoid overfitting

 

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #save only that model for which the validation loss decreased from previous iterations
    callbacks = [checkpoint] #track the checkpoints for model updation
    history = model.fit(X_train, y_train , batch_size = batch_size, validation_data=(X_val, y_val),verbose=True, epochs=epoch, callbacks=[early_stop,callbacks])
 
    # training the model
    model = load_model(filepath) # update the model using the model saved in defined filepath through callbacks

    # Print the best parameters found during random search
    # print("Best Parameters: ", random_search.best_params_)

    return model, history

