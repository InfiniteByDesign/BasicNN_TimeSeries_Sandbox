"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Create a RNN, train using the supplied dataset then test and display the test results on a chart
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
import RNN_Definition as RNN_def

func.print_header()

# Explicitly create a Graph object
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    
    print("Loading Data")
        
    #%% Import the data, specific to the dataset being used for the NN.
    #   Modify this code and the CSVReader to your specific dataset
    x1,x2,x2_Interval,y_Realtime,y_Interval,y_Interval_interpolated,numInputs,numOutputs,trainingSamples,testingSamples = \
        csvreader.Import_CSV(cfg.dir_path, cfg.dir_char,cfg.filename,cfg.trainingPct)

    # Determine the number of samples for testing and training
    trainingSamples = int(len(x1) * cfg.trainingPct / 100)
    testingSamples  = len(x1) - trainingSamples
    
    #%% Pre-process the data    
    x_training_batchs,x_test_batches,                              \
    x_training_NARX_batches,                                       \
    y_Realtime_training_batches,y_Realtime_testing_batches,        \
    y_Interval_training_batches,y_Interval_testing_batches,        \
    y_Interpolate_training_batches,y_Interpolate_testing_batches = \
        func.split_data_into_batches(x1,y_Realtime,y_Interval,y_Interval_interpolated,x2,x2_Interval,trainingSamples,testingSamples,cfg.batchSize,cfg.numInputDelays,cfg.numOutputDelays)

    # Choose from the datasets above for the x and y data
    x_train     = x_training_batchs
    x_test      = x_test_batches
    #y_train     = y_Realtime_training_batches
    #y_test      = y_Realtime_testing_batches
    #y_train     = y_Interval_training_batches
    #y_test      = y_Interval_testing_batches
    y_train     = y_Interpolate_training_batches
    y_test      = y_Interpolate_testing_batches
       
    print("Defining Variables")
       
    # Inputs, Placeholders for the input, output and drop probability
    with tf.name_scope('input'):
        # Input, size determined by batch size and number of inputs per time step
        x = tf.placeholder(tf.float32, shape=[None, cfg.batchSize, numInputs], name="x-input") 
        # Output, size determined by batch size and number of outputs per time step
        y = tf.placeholder(tf.float32, shape=[None, cfg.batchSize, numOutputs], name="y-input")
        # Dropout Keep Pobability
        keep_prob = tf.placeholder("float")

    print("Defining Model")
        
    # Setup the NN Model
    with tf.name_scope('Model'):
        predictions = RNN_def.RNN_create(numInputs, cfg.hidden_layer_widths, numOutputs, cfg.dropout_output_keep_prob, x, cfg.batchSize)

    # The Cost fuction
    with tf.name_scope('cost'):
        cost = tf.reduce_sum(tf.square(predictions - y))
        func.variable_summaries(cost)
    
    # The Optimization algorithm
    with tf.name_scope('optimzer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(cost)        

    # Tensorboard functions and model saving functions
    merged_summary_op = tf.summary.merge_all()                                           
    summary_writer = tf.summary.FileWriter(cfg.dir_path + cfg.log_dir, graph)  
    ckptfile = cfg.dir_path + cfg.dir_char + cfg.model_dir + cfg.dir_char + cfg.model_name
    saver = tf.train.Saver(max_to_keep=1)      

    print("Starting TensorFlow Session")  

    # Train and Test the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print("Training Model")
        
        if cfg.restore==True:
            # Restore the last model
            saver.restore(sess,cfg.dir_path + cfg.dir_char + cfg.model_dir + cfg.dir_char + cfg.restore_File)
        else:
            # Train the Model
            RNN_def.RNN_train(sess,cfg,saver,summary_writer,x,y,x_train,y_train,optimizer,cost,merged_summary_op,ckptfile)
        
        print("Optimization Finished")
        print("Running Test Data...")

        # Test the Model
        pred = RNN_def.RNN_test(sess,cfg,x,x_test,predictions)
        
        # Display the results
        actual = np.reshape(y_Interval_testing_batches,(y_test.shape[0]*y_test.shape[1],y_test.shape[2]))
        interp = np.reshape(y_Interpolate_testing_batches,(y_test.shape[0]*y_test.shape[1],y_test.shape[2]))
        func.plot_test_data(0,actual,pred)
        func.plot_test_data(0,interp,pred)

    # Flushes the summaries to disk and closes the SummaryWriter
    summary_writer.close()