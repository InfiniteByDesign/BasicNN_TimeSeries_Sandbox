"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Create a MLP NN, train using the supplied dataset then test and display the test results on a chart
"""
# Include files
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Include custom files
import CSVReader as csvreader
import functions as func
import Configuration as cfg
import MLP_Definition as mlp

# Explicitly create a Graph object
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
        
    #%% Import the data, specific to the dataset being used for the NN.
    #   Modify this code and the CSVReader to your specific dataset
    molecularWeights,gcRealtime,gcInterval,gcInterval_i,pvs,pvInterval,numInputs,numOutputs,trainingSamples,testingSamples= csvreader.Import_CSV(cfg.filename,cfg.trainingPct)
    
    #%% Pre-process the data    
    x_training_batchs,x_test_batches,y_Realtime_training_batches,y_Realtime_testing_batches = \
        func.split_data_into_batches(molecularWeights,gcRealtime,gcInterval,gcInterval_i,pvs,pvInterval,trainingSamples,testingSamples,cfg.batchSize)

    # Determine the number of samples for testing and training
    trainingSamples = int(len(molecularWeights) * cfg.trainingPct / 100)
    testingSamples  = len(molecularWeights) - trainingSamples
       
    #%% Inputs, Placeholders for the input, output and drop probability
    with tf.name_scope('input'):
        # Input, size determined by batch size and number of inputs per time step
        x = tf.placeholder(tf.float32, shape=[cfg.batchSize, numInputs], name="x-input") 
        # Output, size determined by batch size and number of outputs per time step
        y = tf.placeholder(tf.float32, shape=[cfg.batchSize, numOutputs], name="y-input")
        # Dropout Keep Pobability
        keep_prob = tf.placeholder("float")
        
    #%% Setup the RNN Model
    with tf.name_scope('Model'):

        # Create and initialize the weights of the NN
        with tf.name_scope('weights'):
            weights = {
                'h1': tf.Variable(tf.random_normal([numInputs, cfg.hidden_layer_widths[0]])),
                'out': tf.Variable(tf.random_normal([cfg.hidden_layer_widths[0], numOutputs]))
            }
            with tf.name_scope('h1'):
                func.variable_summaries(weights['h1'])
            with tf.name_scope('out'):
                func.variable_summaries(weights['out'])

        # Create and initialize the biases of the NN
        with tf.name_scope('biases'):
            biases = {
                'b1': tf.Variable(tf.random_normal([cfg.hidden_layer_widths[0]])),
                'out': tf.Variable(tf.random_normal([numOutputs]))
            }
            with tf.name_scope('b1'):
                func.variable_summaries(biases['b1'])
            with tf.name_scope('out'):
                func.variable_summaries(biases['out'])

    # The Prediction function
    with tf.name_scope('predictions'):
        predictions = mlp.multilayer_perceptron(x, weights, biases, keep_prob) 
        func.variable_summaries(predictions)

    # The Cost fuction
    with tf.name_scope('cost'):
        cost = tf.reduce_sum(tf.square(predictions - y))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
        func.variable_summaries(cost)
    
    # The Optimization algorithm
    with tf.name_scope('optimzer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(cost)                            

    # Tensorboard functions and model saving functions
    merged_summary_op = tf.summary.merge_all()                                           
    summary_writer = tf.summary.FileWriter(cfg.dir_path + cfg.log_dir, graph)  
    saver = tf.train.Saver(max_to_keep=1)      

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Train the Model
        mlp.multilayer_perceptron_train(sess,cfg,summary_writer,x,y,keep_prob,x_training_batchs,y_Realtime_training_batches,optimizer,cost,merged_summary_op)
        
        print("Optimization Finished!")
        print("Running Test Data...")

        # Test the Model
        pred = mlp.multiplayer_perceptron_test(sess,cfg,summary_writer,x,keep_prob,x_test_batches,y_Realtime_testing_batches,predictions)
        
        # Display the results
        actual = np.reshape(y_Realtime_testing_batches,(y_Realtime_testing_batches.shape[0]*y_Realtime_testing_batches.shape[1],y_Realtime_testing_batches.shape[2]))
        func.plot_test_data(0,actual,pred)

    # Flushes the summaries to disk and closes the SummaryWriter
    summary_writer.close()