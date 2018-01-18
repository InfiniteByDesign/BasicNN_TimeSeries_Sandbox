import os
import numpy
import pandas as pd

def Import_CSV(File_Name,trainingPct):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + "\\" + File_Name + ".csv"

    # Read realtime gas concentrations
    values = pd.read_csv(filename,usecols = [1,2,3,4,5,6,7,8,9,10], skiprows = [0,1], header=None)
    gcRealtime = numpy.float32(values.values)

    # Read molecular weights
    values = pd.read_csv(filename,usecols = [0], skiprows = [0,1], header=None)
    molecularWeights = numpy.float32(values.values)

    # Read PVs
    values = pd.read_csv(filename,usecols = [21,22,23,24,25,26], skiprows = [0,1], header=None)
    pvs = numpy.float32(values.values)

    # Read interval gas concentrations, interpolate missing data then drop preceding missing data
    values = pd.read_csv(filename,usecols = [11,12,13,14,15,16,17,18,19,20], skiprows = [0,1], header=None)
    gcInterval   = values
    gcInterval_i = values.replace('nan', numpy.NaN)
    gcInterval_i = gcInterval_i.interpolate()
    gcInterval_i = gcInterval_i.dropna(axis=0, how='any')
    gcInterval_i = numpy.float32(gcInterval_i.values)
    
    # Make sure the pv and original gc arrays are only as long as the interpolated array
    l1          = gcInterval_i.shape[0]
    l2          = pvs.shape[0]
    start       = l2-l1-1
    pvInterval  = pvs[start:l2,:]
    gcInterval  = numpy.float32(values.values)[start:l2,:]
    
    
    # Determine the sizes of the data
    numInputs       = pvs.shape[1]
    numOutputs      = gcRealtime.shape[1]
    trainingSamples = int(len(molecularWeights) * trainingPct / 100)
    testingSamples  = len(molecularWeights) - trainingSamples
    
    return molecularWeights,gcRealtime,gcInterval,gcInterval_i,pvs,pvInterval,numInputs,numOutputs,trainingSamples,testingSamples