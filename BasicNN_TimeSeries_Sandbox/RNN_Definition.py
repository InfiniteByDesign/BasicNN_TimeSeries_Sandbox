"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Functions used to create, train, and test a RNN neural network
"""

import datetime
import numpy as np
import tensorflow as tf

#%% Create arrays containing the weights and biases for the MLP
def create_layers(hidden_layer_widths, keep_prob):
    # Create and initialize the weights of the NN
    with tf.name_scope('layers'):
        cells = []
        for i in range(len(hidden_layer_widths)+1):
            # Add hidden layers
            cells[i] = tf.nn.rnn_cell.BasicRNNCell(hidden_layer_widths[i])
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


#%% Train MLP network
def multilayer_perceptron_train(sess,cfg,saver,summary_writer,x,y,keep_prob,x_batches,y_batches,optimizer,cost,merged_summary_op):    
    #  Loop through each epoch
    for epoch in range(cfg.epochs):
        
        summary,_ = sess.run([merged_summary_op, optimizer], feed_dict={x: x_batches, y: y_batches})
        mse = cost.eval(feed_dict={x: x_batches, y: y_batches})
        if epoch % cfg.display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "mse=", "{:.9f}".format(mse))
        
        # Save the model and save the TensorBoard summaries
        save_path = saver.save(sess, cfg.dir_path + cfg.model_dir + cfg.model_name)
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()

    
#%% Test the MLP network
def multiplayer_perceptron_test(sess,cfg,summary_writer,x,keep_prob,x_input,y_shape,predictions):
    # Run the predictions for a batch style input (3 dimensions [batch, timestep, sample])
    if len(x_input.shape)==3:
        predicted = np.zeros(shape=(y_shape[0],y_shape[1],y_shape[2]), dtype=float)
        for i in range(x_input.shape[0]):
            batch_testx = x_input[i,:,:]
            temp = sess.run([predictions], 
                            feed_dict={
                                x: batch_testx,
                                keep_prob: 1.0
                            })
            predicted[i,:,:] = temp[0]
        return np.reshape(predicted,(predicted.shape[0]*predicted.shape[1],y_shape[2]))
    # Run the predictions for a non-batch style input (2 dimensions [timestep, sample])
    elif len(x_input.shape)==2:
        predicted = np.zeros(shape=(y_shape[0],y_shape[1]), dtype=float)
        for i in range(x_input.shape[0]):
            row_x = x_input[i,:]
            row_x = np.reshape(np.append(row_x, row_x,axis=0),[-1,len(row_x)])
            temp = sess.run([predictions], 
                            feed_dict={
                                x: row_x,
                                keep_prob: 1.0
                            })
            predicted[i,:] = temp[0][0,:]
        return predicted
    # Unknown dimension, Error
    else:
        return 0.0

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