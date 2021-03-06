"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Configuration parameters to run the following models
                    Main_NN   - basic multilayer perceptron model
                    Main_NARX - nonlinear autoregressive mlp with exgenous inputs (delayed inputs and outputs fed back into the NN)
"""
import os
import datetime

# The CSV filename for the training/testing data, leave off the .csv
#   InputData1_100k - Sampling Time=1 sec, y Update Time=1 min,  100k samples
#   InputData2_100k - Sampling Time=1 sec, y Update Time=1 min,  100k samples
#   InputData3_100k - Sampling Time=1 sec, y Update Time=15 min, 100k samples
#   InputData3_1M   - Sampling Time=1 sec, y Update Time=15 min, 1M   samples (not provided because of file size)
#   InputData3_10M  - Sampling Time=1 sec, y Update Time=15 min, 10M  samples (not provided because of file size)
filename = "InputData2_100k"


# General NN parameters
trainingPct                 = 90            # The amount of data to use for training
batchSize                   = 500           # Batch size, samples per batch
epochs                      = 200           # Number of iterations or training cycles, includes both the FeedFoward and Backpropogation
learning_rate               = 0.0001        # Learning Rate 
dropout_output_keep_prob    = .75           # Percentage of neurons to keep between 
hidden_layer_widths         = [64, 32, 16]  # List of hidden layer widths (num neurons per hidden layer)
display_step                = 1             # How often to update the console with text
init_weights_bias_mean_val  = 0.0           # Mean value of the normal distribution used to initialize the weights and biases
init_weights_bias_std_dev   = 0.1           # Standard Deviation of the normal distribution used to initialize the weights and biases

# NARX NN parameters (not implemented yet)
numInputDelays              = 5
numOutputDelays             = 20

# Restore Last Model
restore                     = False
restore_File                = 'RNN 2018-01-26 16-39-37.325890.cpkt'

# Path settings, checks for Windows environment to choose between \ and /
if os.name == 'nt':
    dir_char = '\\'
else:
    dir_char = '/'
dir_path = os.path.dirname(os.path.realpath(__file__))
log_dir = dir_char + 'logs'
model_dir = dir_char + 'models'
model_name = dir_char + "RNN " + str(datetime.datetime.now()).replace(":","-") + ".cpkt"