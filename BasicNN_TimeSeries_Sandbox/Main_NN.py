"""
Author: David Beam, db4ai
Date:   18 January 2018
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
        
    #%% Import the data
    molecularWeights,gcRealtime,gcInterval,gcInterval_i,pvs,pvInterval,numInputs,numOutputs,trainingSamples,testingSamples= csvreader.Import_CSV(cfg.filename,cfg.trainingPct)
    
    #%% Pre-process the data    
    # Determine the number of samples for testing and training
    numInputs       = 6
    numOutputs      = 10
    numHidden_RNN   = numInputs
    trainingSamples = int(len(molecularWeights) * cfg.trainingPct / 100)
    testingSamples  = len(molecularWeights) - trainingSamples

    x_input_batches,x_test_batches,gcRealtime_training_batches,gcRealtime_testing_batches,gcRealtime_test = func.split_data_into_batches(molecularWeights,gcRealtime,gcInterval,gcInterval_i,pvs,pvInterval,trainingSamples,testingSamples,cfg.batchSize)
       
#%% Inputs
    # input images
    with tf.name_scope('input'):
        # None -> batch size can be any size, 6 values per time step
        x = tf.placeholder(tf.float32, shape=[cfg.batchSize, 6], name="x-input") 
        # 10 gas concentrations
        y = tf.placeholder(tf.float32, shape=[cfg.batchSize, 10], name="y-input")
        
    #%% Setup the RNN Model
    with tf.name_scope('Model'):
        display_step = 1
        n_hidden_1 = 64
        n_input = x_input_batches.shape[2]
        n_classes = gcRealtime_training_batches.shape[2]

        with tf.name_scope('weights'):
            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
            }
            with tf.name_scope('h1'):
                func.variable_summaries(weights['h1'])
            with tf.name_scope('out'):
                func.variable_summaries(weights['out'])

        with tf.name_scope('biases'):
            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }
            with tf.name_scope('b1'):
                func.variable_summaries(biases['b1'])
            with tf.name_scope('out'):
                func.variable_summaries(biases['out'])

        keep_prob = tf.placeholder("float")
    
    with tf.name_scope('predictions'):
        predictions = mlp.multilayer_perceptron(x, weights, biases, keep_prob) 
        func.variable_summaries(predictions)

    with tf.name_scope('loss'):
        cost = tf.reduce_sum(tf.square(predictions - y))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
        func.variable_summaries(cost)
    
    with tf.name_scope('optimzer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(cost)                            
            
    merged_summary_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=1)                                               
    summary_writer = tf.summary.FileWriter(cfg.dir_path + cfg.log_dir, graph)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        

        # Train the Model
        mlp.multilayer_perceptron_train(sess,cfg,summary_writer,x,y,keep_prob,x_input_batches,gcRealtime_training_batches,optimizer,cost,merged_summary_op)
        
        print("Optimization Finished!")
        print("Running Test Data...")

        # Test the Model
        pred = mlp.multiplayer_perceptron_test(sess,cfg,summary_writer,x,keep_prob,x_test_batches,gcRealtime_testing_batches,predictions)
        
        # Display the results
        actual = np.reshape(gcRealtime_testing_batches,(gcRealtime_testing_batches.shape[0]*gcRealtime_testing_batches.shape[1],gcRealtime_testing_batches.shape[2]))
        func.plot_test_data(0,actual,pred)

    # Flushes the summaries to disk and closes the SummaryWriter
    summary_writer.close()

# tensorboard --logdir="C:\Users\David Beam\Documents\Visual Studio 2017\Projects\PythonApplication1\PythonApplication1\logs"
