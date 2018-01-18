#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 06:22:53 2018

@author: David
"""

import os

# The amount of data to use for training
trainingPct = 90   

# Batch size, samples per batch
batchSize = 100

# Number of iterations or training cycles, includes both the FeedFoward and Backpropogation
epochs = 10

# Learning Rate
learning_rate = 0.001 

# The filename for the training/validating data
filename = "InputData2_100k"

# NARX NN parameters
numInputDelays           = 5
numOutputDelays          = 5

# General NN parameters
dropout_output_keep_prob = .95
num_hidden_layers        = 2
hidden_layer_widths = [64]
display_step = 1                                        # How often to update the console with text

dir_path = os.path.dirname(os.path.realpath(__file__))
log_dir = '\\logs'
model_dir = '\\models'
