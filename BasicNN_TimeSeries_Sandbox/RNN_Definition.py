"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Functions used to create, train, and test a RNN neural network
"""

import datetime
import numpy as np
import tensorflow as tf

#%% Create arrays containing the weights and biases for the MLP
def RNN_create(numInputs, hidden_layer_widths, numOutputs, keep_prob, x, batchSize):
    # Create and initialize the weights of the NN
    with tf.name_scope('layers'):
        cells = []
        for i in range(len(hidden_layer_widths)):
            # Add hidden layers
            last_hidden_width = hidden_layer_widths[i]
            cells.append(tf.nn.rnn_cell.BasicRNNCell(hidden_layer_widths[i]))
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    
        rnn_output, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)                    
        stacked_rnn_output = tf.reshape(rnn_output, [-1, last_hidden_width])                     
        stacked_outputs = tf.layers.dense(stacked_rnn_output, numOutputs)                   
        predictions = tf.reshape(stacked_outputs, [-1, batchSize, numOutputs])     
    return predictions


#%% Load Existing Weights
def RNN_Load_Weights(model,cfg):
    model.load_weights(filepath=cfg.dir_path + cfg.model_dir + cfg.model_name, by_name=False)
    return model


#%% Train MLP network
def RNN_train(sess,cfg,saver,summary_writer,x,y,x_input,y_input,optimizer,cost,merged_summary_op,cpktfile):  
    for ep in range(cfg.epochs):
        summary,_ = sess.run([merged_summary_op, optimizer], feed_dict={x: x_input, y: y_input})
        saver.save(sess, cpktfile)
        mse = cost.eval(feed_dict={x: x_input, y: y_input})
        if ep % cfg.display_step == 0:
            print("Epoch:", '%04d' % (ep+1), "cost=", "{:.9f}".format(mse))
            summary_writer.add_summary(summary, ep)
    

#%% Test the MLP network
def RNN_test(sess,cfg,x,x_input,predictions):
   return sess.run(predictions, feed_dict={x: x_input})


#%% TensorBoard summaries for a given variable
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)