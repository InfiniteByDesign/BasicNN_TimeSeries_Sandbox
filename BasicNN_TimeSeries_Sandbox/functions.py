#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 10:10:55 2018

@author: David
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#%% Create NARX datasets
"""
    Creates a Nonlinear Auto-Regressive w/ Exogenous Input style dataset
    Defined by the number of delayed inputs and the number of delayed outptus
"""
def create_NARX_dataset(input, output, numDelayedInputs, numDelayedOutputs):
    # Calculate the sizes of the data
    numInputs = input.shape[1]
    numOutputs = output.shape[1]
    length = input.shape[0] - max(numDelayedInputs+1,numDelayedOutputs)
    width = ((numDelayedInputs + 1)*numInputs) + (numDelayedOutputs*numOutputs)
    
    # Placeholder to hold the dataset
    x_input_NARX = np.zeros((length, width) , dtype=np.float32)
    
    # Loop through all the inputs
    for i in range(max(numDelayedInputs+1,numDelayedOutputs), input.shape[0]):
        
        # Append delayed inputs to the row        
        temp_row = input[i,:]
        for j in range(1,numDelayedInputs+1):
            temp_row = np.concatenate([temp_row, input[i-j]])
        
        # Append delayed outputs to the row
        for j in range(0,numDelayedOutputs):
            temp_row = np.concatenate([temp_row, output[i-numDelayedOutputs-j,:]], axis=0)
            
        x_input_NARX[i-max(numDelayedInputs+1,numDelayedOutputs),:] = temp_row
    return x_input_NARX


#%% Split the data into training and testing datasets
"""
    Takes the full dataset and splits it into two sets defined by the testing data size and the training data size
"""
def split_data(series,training,testing):
    testing  = series[-testing:]    #split off the testing data
    training = series[0:training]   #split off the training data
    return training,testing


#%% Split data into testing and training sets
"""
    Uses the split_data function to split the required datasets into training and testing sets
"""
def split_data_into_sets(molecularWeights,gcRealtime,gcInterval,gcInterval_i,pvs,pvInterval,trainingSamples,testingSamples,batchSize):
    molW_training,molW_test                 = split_data(molecularWeights,trainingSamples,testingSamples)
    gcRealtime_training,gcRealtime_test     = split_data(gcRealtime,trainingSamples,testingSamples)
    gcInterval_training,gcInterval_test     = split_data(gcInterval,trainingSamples,testingSamples)
    gcInterval_i_training,gcInterval_i_test = split_data(gcInterval_i,trainingSamples,testingSamples)
    pvs_training,pvs_test                   = split_data(pvs,trainingSamples,testingSamples)
    pvInterval_training,pvInterval_test     = split_data(pvInterval,trainingSamples,testingSamples)
    
    return molW_training,molW_test,gcRealtime_training,gcRealtime_test,gcInterval_training,gcInterval_test,gcInterval_i_training,gcInterval_i_test,pvs_training,pvs_test,pvInterval_training,pvInterval_test


#%% Create datasets from the batches
"""
    Take a 2d array and reshapes it into a 3d array with the first dimension being the batch number
    [batch_size, time-step, sample]
"""
def make_batches(series,samples):
    data = series[:(len(series)-(len(series) % samples))]   #trim off extra to ensure equal size batches
    batches = data.reshape(-1, samples, series.shape[1])    #form batches
    return batches


#%% Import the data and separate into batches
def split_data_into_batches(molecularWeights,gcRealtime,gcInterval,gcInterval_i,pvs,pvInterval,trainingSamples,testingSamples,batchSize):
    
    # Split the datasets into testing and training
    molW_training,molW_test,gcRealtime_training,gcRealtime_test,gcInterval_training,gcInterval_test,gcInterval_i_training,gcInterval_i_test,pvs_training,pvs_test,pvInterval_training,pvInterval_test = \
        split_data_into_sets(molecularWeights,gcRealtime,gcInterval,gcInterval_i,pvs,pvInterval,trainingSamples,testingSamples,batchSize)
    
    # Create the input dataset for the RNN model
    x_input = pvs_training
    x_test  = pvs_test
    #x_input = np.concatenate((molW_training, pvs_training), axis=1)
    #x_test  = np.concatenate((molW_test, pvs_test), axis=1)
    
    # Create batches for the RNN model
    gcRealtime_training_batches  = make_batches(gcRealtime_training, batchSize)
    gcRealtime_testing_batches   = make_batches(gcRealtime_test, batchSize)
    x_input_batches              = make_batches(x_input, batchSize)
    x_test_batches               = make_batches(x_test, batchSize)
    
    # Create the input dataset for the NARX model
    #x_input_NARX = create_NARX_dataset(x_input, gcRealtime_training,numInputDelays,numOutputDelays)
    
    return x_input_batches,x_test_batches,gcRealtime_training_batches,gcRealtime_testing_batches


#%% Plot results
def plot_test_data(gas_sample, actual, predict):
    plt.title("Forecast vs Actual, gas " + str(gas_sample), fontsize=14)
    plt.plot(pd.Series(np.ravel(actual[:,gas_sample])), "bo", markersize=1, label="Actual")
    plt.plot(pd.Series(np.ravel(predict[:,gas_sample])), "r.", markersize=1, label="Forecast")
    plt.legend(loc="upper left")
    plt.xlabel("Time Periods")
    plt.show()


#%% TensorBoard summaries for a given variable
def variable_summaries(var):
    #tf.summary.scalar('value',var)
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)