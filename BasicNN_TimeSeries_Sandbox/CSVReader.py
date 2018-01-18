"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Import a .csv file containing data for training and testing a NN
                This function needs to be customized to fit your own data file
"""

import os
import numpy
import pandas as pd

def Import_CSV(dir_path, dir_char, File_Name,trainingPct):
    filename = dir_path + dir_char + File_Name + ".csv"

    # Read realtime gas concentrations
    temp = pd.read_csv(filename,usecols = [1,2,3,4,5,6,7,8,9,10], skiprows = [0,1], header=None)
    y_Realtime = numpy.float32(temp.values)

    # Read molecular weights
    temp = pd.read_csv(filename,usecols = [0], skiprows = [0,1], header=None)
    x1_Realtime = numpy.float32(temp.values)

    # Read PVs
    temp = pd.read_csv(filename,usecols = [21,22,23,24,25,26], skiprows = [0,1], header=None)
    x2_Realtime = numpy.float32(temp.values)

    # Read interval gas concentrations, interpolate missing data then drop preceding missing data
    temp = pd.read_csv(filename,usecols = [11,12,13,14,15,16,17,18,19,20], skiprows = [0,1], header=None)
    y_Interval_Interpolated = temp.replace('nan', numpy.NaN)
    y_Interval_Interpolated = y_Interval_Interpolated.interpolate()
    y_Interval_Interpolated = y_Interval_Interpolated.dropna(axis=0, how='any')
    y_Interval_Interpolated = numpy.float32(y_Interval_Interpolated.values)
    
    # Make sure the pv and original gc arrays are only as long as the interpolated array
    l1          = y_Interval_Interpolated.shape[0]
    l2          = x2_Realtime.shape[0]
    start       = l2-l1-1
    x2_Interval = x2_Realtime[start:l2,:]
    y_Interval  = numpy.float32(temp.values)[start:l2,:]
    
    
    # Determine the sizes of the data
    numInputs       = x2_Realtime.shape[1]
    numOutputs      = y_Realtime.shape[1]
    trainingSamples = int(len(x1_Realtime) * trainingPct / 100)
    testingSamples  = len(x1_Realtime) - trainingSamples
    
    return x1_Realtime,x2_Realtime,x2_Interval,y_Realtime,y_Interval,y_Interval_Interpolated,numInputs,numOutputs,trainingSamples,testingSamples